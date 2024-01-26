/* This example expose parameter to pass generator of sample.
Good starting point for integration of cpal into your application.
*/

extern crate anyhow;
extern crate clap;
extern crate cpal;

use std::sync::Arc;

use rand::prelude::*;
// use std::{slice::Chunks, sync::Arc};

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    SizedSample,
};
use cpal::{FromSample, Sample};

fn main() -> anyhow::Result<()> {
    let stream = stream_setup_for()?;
    stream.play()?;
    std::thread::sleep(std::time::Duration::from_millis(4000));
    Ok(())
}

// pub enum Waveform {
//     Sine,
//     Square,
//     Saw,
//     Triangle,
// }

// pub struct Oscillator {
//     pub sample_rate: f32,
//     pub waveform: Waveform,
//     pub current_sample_index: f32,
//     pub frequency_hz: f32,
// }

// impl Oscillator {
//     fn advance_sample(&mut self) {
//         self.current_sample_index = (self.current_sample_index + 1.0) % self.sample_rate;
//     }

//     fn set_waveform(&mut self, waveform: Waveform) {
//         self.waveform = waveform;
//     }

//     fn calculate_sine_output_from_freq(&self, freq: f32) -> f32 {
//         let two_pi = 2.0 * std::f32::consts::PI;
//         (self.current_sample_index * freq * two_pi / self.sample_rate).sin()
//     }

//     fn is_multiple_of_freq_above_nyquist(&self, multiple: f32) -> bool {
//         self.frequency_hz * multiple > self.sample_rate / 2.0
//     }

//     fn sine_wave(&mut self) -> f32 {
//         self.advance_sample();
//         self.calculate_sine_output_from_freq(self.frequency_hz)
//     }

//     fn generative_waveform(&mut self, harmonic_index_increment: i32, gain_exponent: f32) -> f32 {
//         self.advance_sample();
//         let mut output = 0.0;
//         let mut i = 1;
//         while !self.is_multiple_of_freq_above_nyquist(i as f32) {
//             let gain = 1.0 / (i as f32).powf(gain_exponent);
//             output += gain * self.calculate_sine_output_from_freq(self.frequency_hz * i as f32);
//             i += harmonic_index_increment;
//         }
//         output
//     }

//     fn square_wave(&mut self) -> f32 {
//         self.generative_waveform(2, 1.0)
//     }

//     fn saw_wave(&mut self) -> f32 {
//         self.generative_waveform(1, 1.0)
//     }

//     fn triangle_wave(&mut self) -> f32 {
//         self.generative_waveform(2, 2.0)
//     }

//     fn tick(&mut self) -> f32 {
//         match self.waveform {
//             Waveform::Sine => self.sine_wave(),
//             Waveform::Square => self.square_wave(),
//             Waveform::Saw => self.saw_wave(),
//             Waveform::Triangle => self.triangle_wave(),
//         }
//     }
// }

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
    // let mut oscillator = Oscillator {
    //     waveform: Waveform::Sine,
    //     sample_rate: config.sample_rate.0 as f32,
    //     current_sample_index: 0.0,
    //     frequency_hz: 440.0,
    // };
    let err_fn = |err| eprintln!("Error building output sound stream: {}", err);

    let time_at_start = std::time::Instant::now();
    println!("Time at start: {:?}", time_at_start);

    // TODO make size real
    let mut lenia = Lenia::new(0.5, 512);

    let stream = device.build_output_stream(
        config,
        move |output: &mut [T], _: &cpal::OutputCallbackInfo| {
            // for 0-1s play sine, 1-2s play square, 2-3s play saw, 3-4s play triangle_wave
            // let time_since_start = std::time::Instant::now()
            //     .duration_since(time_at_start)
            //     .as_secs_f32();
            // if time_since_start < 1.0 {
            //     oscillator.set_waveform(Waveform::Sine);
            // } else if time_since_start < 2.0 {
            //     oscillator.set_waveform(Waveform::Triangle);
            // } else if time_since_start < 3.0 {
            //     oscillator.set_waveform(Waveform::Square);
            // } else if time_since_start < 4.0 {
            //     oscillator.set_waveform(Waveform::Saw);
            // } else {
            //     oscillator.set_waveform(Waveform::Sine);
            // }

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
    lenia.step();
    let world = &mut lenia.world;
    println!("world: {:?}", world);
    let new: Vec<SampleType> = world
        .iter_mut()
        .map(|s| SampleType::from_sample(*s))
        .collect();
    for frame in output.chunks_mut(num_channels) {
        // let value: SampleType = SampleType::from_sample(oscillator.tick());

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
        let forward: Arc<dyn RealToComplex<f32>> = planner.plan_fft_forward(size);
        let inverse: Arc<dyn ComplexToReal<f32>> = planner.plan_fft_inverse(size);
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

    pub fn step(&mut self) {
        let mut conv = self.conv();

        for i in 0..self.size {
            let mut c = conv[i];
            c = Lenia::growth(c);

            let t = c * self.factor;
            if t < 0.0 {
                conv[i] = 0.0;
            } else if t > 1.0 {
                conv[i] = 1.0;
            } else {
                conv[i] = t;
            }
        }

        self.world = conv;
    }

    fn conv(&mut self) -> Vec<f32> {
        let world = &self.world;

        let chunk_size = self.size / 4;
        let mut chunk_1 = Vec::from(&world[0..chunk_size]);
        let mut chunk_2 = Vec::from(&world[chunk_size..(chunk_size * 2)]);
        let mut chunk_3 = Vec::from(&world[(chunk_size * 2)..(chunk_size * 3)]);

        let zeros: Vec<f32> = vec![0.0; chunk_size];
        chunk_1.extend(&zeros);
        chunk_2.extend(&zeros);
        chunk_3.extend(&zeros);

        let kernel_slice = self.kernel.as_mut_slice();
        let mut kernel_freq: Vec<Complex<f32>> = self.forward_fft.make_output_vec();
        let _ = self
            .forward_fft
            .process(kernel_slice, kernel_freq.as_mut_slice());

        let mut chunk_1_freq: Vec<Complex<f32>> = self.forward_fft.make_output_vec();
        let mut chunk_2_freq: Vec<Complex<f32>> = self.forward_fft.make_output_vec();
        let mut chunk_3_freq: Vec<Complex<f32>> = self.forward_fft.make_output_vec();

        let _ = self
            .forward_fft
            .process(chunk_1.as_mut_slice(), chunk_1_freq.as_mut_slice());

        let _ = self
            .forward_fft
            .process(chunk_2.as_mut_slice(), chunk_2_freq.as_mut_slice());

        let _ = self
            .forward_fft
            .process(chunk_3.as_mut_slice(), chunk_3_freq.as_mut_slice());

        let mut conv_1 = self.inverse_fft.make_output_vec();
        let mut conv_1_freq: Vec<Complex<f32>> = self.inverse_fft.make_input_vec();
        let mut conv_2 = self.inverse_fft.make_output_vec();
        let mut conv_2_freq: Vec<Complex<f32>> = self.inverse_fft.make_input_vec();
        let mut conv_3 = self.inverse_fft.make_output_vec();
        let mut conv_3_freq: Vec<Complex<f32>> = self.inverse_fft.make_input_vec();

        // TODO figure out how exactly to do this multiply;
        for i in 0..(self.size / 2) {
            conv_1_freq[i] = chunk_1_freq[i] * kernel_freq[i];
            conv_2_freq[i] = chunk_2_freq[i] * kernel_freq[i];
            conv_3_freq[i] = chunk_3_freq[i] * kernel_freq[i];
        }

        let _ = self
            .inverse_fft
            .process(conv_1_freq.as_mut_slice(), conv_1.as_mut_slice());

        let _ = self
            .inverse_fft
            .process(conv_2_freq.as_mut_slice(), conv_2.as_mut_slice());

        let _ = self
            .inverse_fft
            .process(conv_3_freq.as_mut_slice(), conv_3.as_mut_slice());

        // TODO this is really not fast (and its ugly) (and gross ew)
        let mut conv: Vec<f32> = vec![0.0; self.size];
        for i in 0..(self.size / 2) {
            conv[i] = conv_1[i];
        }
        for i in 0..(self.size / 2) {
            conv[i + (self.size / 4)] += conv_2[i];
        }
        for i in 0..(self.size / 2) {
            conv[i + (self.size / 2)] += conv_3[i];
        }

        return conv;
    }

    fn growth(g: f32) -> f32 {
        2.0 * f32::exp2(((g - 0.33).powf(2.0)) / -0.01) - 1.0
    }
}
