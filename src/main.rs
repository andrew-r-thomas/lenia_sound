/* This example expose parameter to pass generator of sample.
Good starting point for integration of cpal into your application.
*/
// TODO add a ringbuf so that we dont have to care how large
// the chunks coming in are

extern crate anyhow;
extern crate clap;
extern crate cpal;

use std::sync::Arc;

use rand::prelude::*;

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    SizedSample,
};
use cpal::{FromSample, Sample};
use ringbuf::StaticRb;

fn main() -> anyhow::Result<()> {
    let stream = stream_setup_for()?;
    stream.play()?;
    std::thread::sleep(std::time::Duration::from_millis(10000));
    Ok(())
}

pub fn stream_setup_for() -> Result<cpal::Stream, anyhow::Error>
where
{
    let (_host, device, config) = host_device_setup()?;

    match config.sample_format() {
        cpal::SampleFormat::I8 => make_stream::<i8>(&device, &config.into()),
        cpal::SampleFormat::I16 => make_stream::<i16>(&device, &config.into()),
        cpal::SampleFormat::I32 => make_stream::<i32>(&device, &config.into()),
        cpal::SampleFormat::I64 => make_stream::<i64>(&device, &config.into()),
        cpal::SampleFormat::U8 => make_stream::<u8>(&device, &config.into()),
        cpal::SampleFormat::U16 => make_stream::<u16>(&device, &config.into()),
        cpal::SampleFormat::U32 => make_stream::<u32>(&device, &config.into()),
        cpal::SampleFormat::U64 => make_stream::<u64>(&device, &config.into()),
        cpal::SampleFormat::F32 => make_stream::<f32>(&device, &config.into()),
        cpal::SampleFormat::F64 => make_stream::<f64>(&device, &config.into()),
        sample_format => Err(anyhow::Error::msg(format!(
            "Unsupported sample format '{sample_format}'"
        ))),
    }
}

pub fn host_device_setup(
) -> Result<(cpal::Host, cpal::Device, cpal::SupportedStreamConfig), anyhow::Error> {
    let host = cpal::default_host();

    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow::Error::msg("Default output device is not available"))?;
    println!("Output device : {}", device.name()?);

    let config = device.default_output_config()?;
    println!("Default output config : {:?}", config);

    // TODO make this more elegant
    // let configured = StreamConfig {
    //     channels: config.channels(),
    //     sample_rate: config.sample_rate(),
    //     buffer_size: cpal::BufferSize::Fixed(512),
    // };

    Ok((host, device, config))
}

pub fn make_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
) -> Result<cpal::Stream, anyhow::Error>
where
    T: SizedSample + FromSample<f32>,
{
    let num_channels = config.channels as usize;

    let err_fn = |err| eprintln!("Error building output sound stream: {}", err);

    let time_at_start = std::time::Instant::now();
    println!("Time at start: {:?}", time_at_start);

    // TODO make these types a little better
    // let mut rb = StaticRb::<f32, 1024>::default();
    // let (mut prod, mut cons) = rb.split_ref();

    // TODO make size real
    let mut lenia = Lenia::new(0.5, 512);

    let stream = device.build_output_stream(
        config,
        move |output: &mut [T], _: &cpal::OutputCallbackInfo| {
            process_frame(output, &mut lenia, num_channels)
        },
        err_fn,
        None,
    )?;

    Ok(stream)
}

fn process_frame<SampleType>(
    output: &mut [SampleType],
    lenia: &mut Lenia,
    // oscillator: &mut Oscillator,
    num_channels: usize,
) where
    SampleType: Sample + FromSample<f32>,
{
    let mut world = lenia.step();
    let new: Vec<SampleType> = world
        .iter_mut()
        .map(|s| SampleType::from_sample(*s))
        .collect();
    for frame in output.chunks_mut(num_channels) {
        // copy the same value to all channels
        let mut i = 0;
        for sample in frame.iter_mut() {
            *sample = new[i];
            i += 1;
        }
    }
}

pub struct Lenia {
    world: Vec<f32>,
    kernel: Vec<f32>,
    factor: f32,
    size: usize,
    forward_fft: Arc<dyn RealToComplex<f32>>,
    inverse_fft: Arc<dyn ComplexToReal<f32>>,
}

impl Lenia {
    pub fn new(factor: f32, size: usize) -> Lenia {
        let mut rng = rand::thread_rng();
        let mut k = vec![0.0; size / 4];
        for i in 0..(k.len() / 2) {
            if i == 0 {
                continue;
            }
            let x: f32 = i as f32 / (k.len() / 2) as f32;

            // TODO make kernel more configurable
            let val = f32::exp2(4.0 - (4.0 / (4.0 * x * (1.0 - x))));

            k[i] = val;
            k[i + ((size / 4) / 2)] = val;
        }

        let mut planner = RealFftPlanner::<f32>::new();
        let forward: Arc<dyn RealToComplex<f32>> = planner.plan_fft_forward(size / 2);
        let inverse: Arc<dyn ComplexToReal<f32>> = planner.plan_fft_inverse(size / 2);
        let world = (0..size).map(|_| rng.gen()).collect();

        return Lenia {
            world,
            kernel: k,
            factor,
            size,
            forward_fft: forward,
            inverse_fft: inverse,
        };
    }

    pub fn step(&mut self) -> Vec<f32> {
        // TODO probably dont need to be cloning all this stuff
        let world = self.world.clone();
        println!("world during step: {:?}", world);
        let kernel = self.kernel.clone();
        let forward_fft = self.forward_fft.clone();
        let inverse_fft = self.inverse_fft.clone();

        let mut conv = Lenia::conv(&world, kernel, self.size / 4, forward_fft, inverse_fft);

        for i in 0..self.size {
            let mut c = conv[i];
            c = Lenia::growth(c);
            let w = world[i];

            let t = c * self.factor;
            let new = w + t;
            if new < 0.0 {
                conv[i] = 0.0;
            } else if new > 1.0 {
                conv[i] = 1.0;
            } else {
                conv[i] = new;
            }
        }

        self.world = conv.clone();
        conv
    }

    fn conv(
        world: &Vec<f32>,
        mut kernel: Vec<f32>,
        chunk_size: usize,
        forward_fft: Arc<dyn RealToComplex<f32>>,
        inverse_fft: Arc<dyn ComplexToReal<f32>>,
    ) -> Vec<f32> {
        let mut chunk_1 = Vec::from(&world[0..chunk_size]);
        let mut chunk_2 = Vec::from(&world[chunk_size..(chunk_size * 2)]);
        let mut chunk_3 = Vec::from(&world[(chunk_size * 2)..(chunk_size * 3)]);

        let zeros: Vec<f32> = vec![0.0; chunk_size];
        chunk_1.extend(&zeros);
        chunk_2.extend(&zeros);
        chunk_3.extend(&zeros);

        let mut kernel_freq: Vec<Complex<f32>> = forward_fft.make_output_vec();
        kernel.extend(&zeros);
        let _ = forward_fft.process(&mut kernel, &mut kernel_freq).unwrap();

        let mut chunk_1_freq: Vec<Complex<f32>> = forward_fft.make_output_vec();
        let mut chunk_2_freq: Vec<Complex<f32>> = forward_fft.make_output_vec();
        let mut chunk_3_freq: Vec<Complex<f32>> = forward_fft.make_output_vec();

        let _ = forward_fft
            .process(&mut chunk_1, &mut chunk_1_freq)
            .unwrap();

        let _ = forward_fft
            .process(&mut chunk_2, &mut chunk_2_freq)
            .unwrap();

        let _ = forward_fft
            .process(&mut chunk_3, &mut chunk_3_freq)
            .unwrap();

        let mut conv_1 = inverse_fft.make_output_vec();
        let mut conv_1_freq: Vec<Complex<f32>> = inverse_fft.make_input_vec();
        let mut conv_2 = inverse_fft.make_output_vec();
        let mut conv_2_freq: Vec<Complex<f32>> = inverse_fft.make_input_vec();
        let mut conv_3 = inverse_fft.make_output_vec();
        let mut conv_3_freq: Vec<Complex<f32>> = inverse_fft.make_input_vec();

        for i in 0..(chunk_size) {
            conv_1_freq[i] = chunk_1_freq[i] * kernel_freq[i];
            conv_2_freq[i] = chunk_2_freq[i] * kernel_freq[i];
            conv_3_freq[i] = chunk_3_freq[i] * kernel_freq[i];
        }

        let _ = inverse_fft.process(&mut conv_1_freq, &mut conv_1);

        let _ = inverse_fft.process(&mut conv_2_freq, &mut conv_2);

        let _ = inverse_fft.process(&mut conv_3_freq, &mut conv_3);

        // TODO this is really not fast (and its ugly) (and gross ew)
        let mut conv: Vec<f32> = vec![0.0; chunk_size * 4];
        for i in 0..(chunk_size * 2) {
            conv[i] = conv_1[i];
        }
        for i in 0..(chunk_size * 2) {
            conv[i + (chunk_size * 2)] += conv_2[i];
        }
        for i in 0..(chunk_size * 2) {
            conv[i + (chunk_size * 2)] += conv_3[i];
        }

        return conv;
    }

    fn growth(g: f32) -> f32 {
        let step1 = g - 0.33;
        let step2 = step1 * step1;
        let step3 = step2 / -0.01;
        let step4 = f32::exp(step3);
        let step5 = step4 * 2.0;
        step5 - 1.0
    }
}
