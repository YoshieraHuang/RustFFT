extern crate criterion;
extern crate rustfft;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use std::sync::Arc;
use rustfft::FFT;
use rustfft::num_complex::Complex;
use rustfft::algorithm::*;
use rustfft::algorithm::butterflies::*;

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length
fn bench_fft(c: &mut Criterion, len: usize, id: &str) {

    let mut planner = rustfft::FFTplanner::new(false);
    let fft = planner.plan_fft(len);

    let mut signal = vec![Complex{re: 0_f32, im: 0_f32}; len];
    let mut spectrum = signal.clone();
    c.bench_function(id, |b| b.iter(|| {fft.process(&mut signal, &mut spectrum);} ));
}


// Powers of 4
fn complex_p2_00000064(c: &mut Criterion) { bench_fft(c,       64, "complex_p2_00000064"); }
fn complex_p2_00000256(c: &mut Criterion) { bench_fft(c,      256, "complex_p2_00000256"); }
fn complex_p2_00001024(c: &mut Criterion) { bench_fft(c,     1024, "complex_p2_00001024"); }
fn complex_p2_00004096(c: &mut Criterion) { bench_fft(c,     4096, "complex_p2_00004096"); }
fn complex_p2_00016384(c: &mut Criterion) { bench_fft(c,    16384, "complex_p2_00016384"); }
fn complex_p2_00065536(c: &mut Criterion) { bench_fft(c,    65536, "complex_p2_00065536"); }
fn complex_p2_01048576(c: &mut Criterion) { bench_fft(c,  1048576, "complex_p2_01048576"); }
fn complex_p2_16777216(c: &mut Criterion) { bench_fft(c, 16777216, "complex_p2_16777216"); }
criterion_group!(complex_p2, complex_p2_00000064, complex_p2_00000256, complex_p2_00001024, complex_p2_00004096,
                             complex_p2_00016384, complex_p2_00065536, complex_p2_01048576, complex_p2_16777216);


// Powers of 7
fn complex_p7_00343(c: &mut Criterion) { bench_fft(c,   343, "complex_p7_00343"); }
fn complex_p7_02401(c: &mut Criterion) { bench_fft(c,  2401, "complex_p7_02401"); }
fn complex_p7_16807(c: &mut Criterion) { bench_fft(c, 16807, "complex_p7_16807"); }

criterion_group!(complex_p7, complex_p7_00343, complex_p7_02401, complex_p7_16807);

// Prime lengths
fn complex_prime_000005(c: &mut Criterion) { bench_fft(c,      5, "complex_prime_000005"); }
fn complex_prime_000017(c: &mut Criterion) { bench_fft(c,     17, "complex_prime_000017"); }
fn complex_prime_000151(c: &mut Criterion) { bench_fft(c,    151, "complex_prime_000151"); }
fn complex_prime_000257(c: &mut Criterion) { bench_fft(c,    257, "complex_prime_000257"); }
fn complex_prime_001009(c: &mut Criterion) { bench_fft(c,   1009, "complex_prime_001009"); }
fn complex_prime_002017(c: &mut Criterion) { bench_fft(c,   2017, "complex_prime_002017"); }
fn complex_prime_065537(c: &mut Criterion) { bench_fft(c,  65537, "complex_prime_065537"); }
fn complex_prime_746497(c: &mut Criterion) { bench_fft(c, 746497, "complex_prime_746497"); }

criterion_group!(complex_prime, complex_prime_000005, complex_prime_000017, complex_prime_000151, complex_prime_000257,
                                complex_prime_001009, complex_prime_002017, complex_prime_065537, complex_prime_746497);

//primes raised to a power
fn complex_primepower_044521(c: &mut Criterion) { bench_fft(c,  44521, "complex_primepower_044521"); } // 211^2
fn complex_primepower_160801(c: &mut Criterion) { bench_fft(c, 160801, "complex_primepower_160801"); } // 401^2

criterion_group!(complex_primepower, complex_primepower_044521, complex_primepower_160801);

// numbers times powers of two
fn complex_composite_24576(c: &mut Criterion) { bench_fft(c,  24576, "complex_composite_24576"); }
fn complex_composite_20736(c: &mut Criterion) { bench_fft(c,  20736, "complex_composite_20736"); }

// power of 2 times large prime
fn complex_composite_32192(c: &mut Criterion) { bench_fft(c,  32192, "complex_composite_32192"); }
fn complex_composite_24028(c: &mut Criterion) { bench_fft(c,  24028, "complex_composite_24028"); }

// small mixed composites times a large prime
fn complex_composite_30270(c: &mut Criterion) { bench_fft(c,  30270, "complex_composite_30270"); }

criterion_group!(complex_composite, complex_composite_20736, complex_composite_24028, complex_composite_24576, complex_composite_30270, complex_composite_32192);

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to the Good-Thomas algorithm
fn bench_good_thomas(c: &mut Criterion, width: usize, height: usize, id: &str) {

    let mut planner = rustfft::FFTplanner::new(false);
    let width_fft = planner.plan_fft(width);
    let height_fft = planner.plan_fft(height);

    let fft : Arc<dyn FFT<_>> = Arc::new(GoodThomasAlgorithm::new(width_fft, height_fft));

    let mut signal = vec![Complex{re: 0_f32, im: 0_f32}; width * height];
    let mut spectrum = signal.clone();
    c.bench_function(id, |b| b.iter(|| {fft.process(&mut signal, &mut spectrum);} ));
}

fn good_thomas_0002_3(c: &mut Criterion) { bench_good_thomas(c,  2, 3, "good_thomas_0002_3"); }
fn good_thomas_0003_4(c: &mut Criterion) { bench_good_thomas(c,  3, 4, "good_thomas_0003_4"); }
fn good_thomas_0004_5(c: &mut Criterion) { bench_good_thomas(c,  4, 5, "good_thomas_0004_5"); }
fn good_thomas_0007_32(c: &mut Criterion) { bench_good_thomas(c, 7, 32, "good_thomas_0007_32"); }
fn good_thomas_0032_27(c: &mut Criterion) { bench_good_thomas(c,  32, 27, "good_thomas_00032_27"); }
fn good_thomas_0256_243(c: &mut Criterion) { bench_good_thomas(c,  256, 243, "good_thomas_0256_243"); }
fn good_thomas_2048_3(c: &mut Criterion) { bench_good_thomas(c,  2048, 3, "good_thomas_2048_3"); }
fn good_thomas_2048_2187(c: &mut Criterion) { bench_good_thomas(c,  2048, 2187, "good_thomas_0002_2187"); }

criterion_group!(good_thomas, good_thomas_0002_3, good_thomas_0003_4, good_thomas_0004_5, good_thomas_0007_32,
                              good_thomas_0032_27, good_thomas_0256_243, good_thomas_2048_2187, good_thomas_2048_3,
                              good_thomas_2048_2187);

/// Times just the FFT setup (not execution)
/// for a given length, specific to the Good-Thomas algorithm
fn bench_good_thomas_setup(c: &mut Criterion, width: usize, height: usize, id: &str) {

    let mut planner = rustfft::FFTplanner::new(false);
    let width_fft = planner.plan_fft(width);
    let height_fft = planner.plan_fft(height);

    c.bench_function(id, |b| b.iter(|| { 
        let fft : Arc<dyn FFT<f32>> = Arc::new(GoodThomasAlgorithm::new(Arc::clone(&width_fft), Arc::clone(&height_fft)));
        black_box(fft);
    }));
}

fn good_thomas_setup_0002_3(c: &mut Criterion) { bench_good_thomas_setup(c,  2, 3, "good_thomas_setup_0002_3"); }
fn good_thomas_setup_0003_4(c: &mut Criterion) { bench_good_thomas_setup(c,  3, 4, "good_thomas_setup_0003_4"); }
fn good_thomas_setup_0004_5(c: &mut Criterion) { bench_good_thomas_setup(c,  4, 5, "good_thomas_setup_0004_5"); }
fn good_thomas_setup_0007_32(c: &mut Criterion) { bench_good_thomas_setup(c, 7, 32, "good_thomas_setup_0007_32"); }
fn good_thomas_setup_0032_27(c: &mut Criterion) { bench_good_thomas_setup(c,  32, 27, "good_thomas_setup_0032_27"); }
fn good_thomas_setup_0256_243(c: &mut Criterion) { bench_good_thomas_setup(c,  256, 243, "good_thomas_setup_0256_243"); }
fn good_thomas_setup_2048_3(c: &mut Criterion) { bench_good_thomas_setup(c,  2048, 3, "good_thomas_setup_2048_3"); }
fn good_thomas_setup_2048_2187(c: &mut Criterion) { bench_good_thomas_setup(c,  2048, 2187, "good_thomas_setup_2048_2187"); }

criterion_group!(good_thomas_setup, good_thomas_setup_0002_3, good_thomas_setup_0003_4, good_thomas_setup_0004_5, good_thomas_setup_0007_32,
                                    good_thomas_setup_0032_27, good_thomas_setup_0256_243, good_thomas_setup_2048_3, good_thomas_setup_2048_2187);

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to the Mixed-Radix algorithm
fn bench_mixed_radix(c: &mut Criterion, width: usize, height: usize, id: &str) {

    let mut planner = rustfft::FFTplanner::new(false);
    let width_fft = planner.plan_fft(width);
    let height_fft = planner.plan_fft(height);

    let fft : Arc<dyn FFT<_>> = Arc::new(MixedRadix::new(width_fft, height_fft));

    let mut signal = vec![Complex{re: 0_f32, im: 0_f32}; width * height];
    let mut spectrum = signal.clone();
    c.bench_function(id, |b| b.iter(|| {fft.process(&mut signal, &mut spectrum);} ));
}

fn mixed_radix_0002_3(c: &mut Criterion) { bench_mixed_radix(c,  2, 3, "mixed_radix_0002_3"); }
fn mixed_radix_0003_4(c: &mut Criterion) { bench_mixed_radix(c,  3, 4, "mixed_radix_0003_4"); }
fn mixed_radix_0004_5(c: &mut Criterion) { bench_mixed_radix(c,  4, 5, "mixed_radix_0004_5"); }
fn mixed_radix_0007_32(c: &mut Criterion) { bench_mixed_radix(c, 7, 32, "mixed_radix_0007_32"); }
fn mixed_radix_0032_27(c: &mut Criterion) { bench_mixed_radix(c,  32, 27, "mixed_radix_0032_27"); }
fn mixed_radix_0256_243(c: &mut Criterion) { bench_mixed_radix(c,  256, 243, "mixed_radix_0256_243"); }
fn mixed_radix_2048_3(c: &mut Criterion) { bench_mixed_radix(c,  2048, 3, "mixed_radix_2048_3"); }
fn mixed_radix_2048_2187(c: &mut Criterion) { bench_mixed_radix(c,  2048, 2187, "mixed_radix_2048_2187"); }

criterion_group!(mixed_radix, mixed_radix_0002_3, mixed_radix_0003_4, mixed_radix_0004_5, mixed_radix_0007_32,
                              mixed_radix_0032_27, mixed_radix_0256_243, mixed_radix_2048_3, mixed_radix_2048_2187);

fn plan_butterfly(len: usize) -> Arc<dyn FFTButterfly<f32>> {
        match len {
            2 => Arc::new(Butterfly2::new(false)),
            3 => Arc::new(Butterfly3::new(false)),
            4 => Arc::new(Butterfly4::new(false)),
            5 => Arc::new(Butterfly5::new(false)),
            6 => Arc::new(Butterfly6::new(false)),
            7 => Arc::new(Butterfly7::new(false)),
            8 => Arc::new(Butterfly8::new(false)),
            16 => Arc::new(Butterfly16::new(false)),
            32 => Arc::new(Butterfly32::new(false)),
            _ => panic!("Invalid butterfly size: {}", len),
        }
    }

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to the Mixed-Radix Double Butterfly algorithm
fn bench_mixed_radix_butterfly(c: &mut Criterion, width: usize, height: usize, id: &str) {

    let width_fft = plan_butterfly(width);
    let height_fft = plan_butterfly(height);

    let fft : Arc<dyn FFT<_>> = Arc::new(MixedRadixDoubleButterfly::new(width_fft, height_fft));

    let mut signal = vec![Complex{re: 0_f32, im: 0_f32}; width * height];
    let mut spectrum = signal.clone();
    c.bench_function(id, |b| b.iter(|| {fft.process(&mut signal, &mut spectrum);} ));
}

fn mixed_radix_butterfly_0002_3(c: &mut Criterion) { bench_mixed_radix_butterfly(c,  2, 3, "mixed_radix_butterfly_0002_3"); }
fn mixed_radix_butterfly_0003_4(c: &mut Criterion) { bench_mixed_radix_butterfly(c,  3, 4, "mixed_radix_butterfly_0003_4"); }
fn mixed_radix_butterfly_0004_5(c: &mut Criterion) { bench_mixed_radix_butterfly(c,  4, 5, "mixed_radix_butterfly_0004_5"); }
fn mixed_radix_butterfly_0007_32(c: &mut Criterion) { bench_mixed_radix_butterfly(c, 7, 32, "mixed_radix_butterfly_0007_32"); }

criterion_group!(mixed_radix_butterfly, mixed_radix_butterfly_0002_3, mixed_radix_butterfly_0003_4, mixed_radix_butterfly_0004_5, mixed_radix_butterfly_0007_32);

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to the Mixed-Radix Double Butterfly algorithm
fn bench_good_thomas_butterfly(c: &mut Criterion, width: usize, height: usize, id: &str) {

    let width_fft = plan_butterfly(width);
    let height_fft = plan_butterfly(height);

    let fft : Arc<dyn FFT<_>> = Arc::new(GoodThomasAlgorithmDoubleButterfly::new(width_fft, height_fft));

    let mut signal = vec![Complex{re: 0_f32, im: 0_f32}; width * height];
    let mut spectrum = signal.clone();
    c.bench_function(id, |b| b.iter(|| {fft.process(&mut signal, &mut spectrum);} ));
}

fn good_thomas_butterfly_0002_3(c: &mut Criterion) { bench_good_thomas_butterfly(c,  2, 3, "good_thomas_butterfly_0002_3"); }
fn good_thomas_butterfly_0003_4(c: &mut Criterion) { bench_good_thomas_butterfly(c,  3, 4, "good_thomas_butterfly_0003_4"); }
fn good_thomas_butterfly_0004_5(c: &mut Criterion) { bench_good_thomas_butterfly(c,  4, 5, "good_thomas_butterfly_0004_5"); }
fn good_thomas_butterfly_0007_32(c: &mut Criterion) { bench_good_thomas_butterfly(c, 7, 32, "good_thomas_butterfly_0007_32"); }

criterion_group!(good_thomas_butterfly, good_thomas_butterfly_0002_3, good_thomas_butterfly_0003_4, good_thomas_butterfly_0004_5, good_thomas_butterfly_0007_32);

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length, specific to Rader's algorithm
fn bench_raders(c: &mut Criterion, len: usize, id: &str) {

    let mut planner = rustfft::FFTplanner::new(false);
    let inner_fft = planner.plan_fft(len - 1);

    let fft : Arc<dyn FFT<_>> = Arc::new(RadersAlgorithm::new(len, inner_fft));

    let mut signal = vec![Complex{re: 0_f32, im: 0_f32}; len];
    let mut spectrum = signal.clone();
    c.bench_function(id,|b| b.iter(|| {fft.process(&mut signal, &mut spectrum);} ));
}

fn raders_000005(c: &mut Criterion) { bench_raders(c,     5, "raders_000005"); }
fn raders_000017(c: &mut Criterion) { bench_raders(c,    17, "raders_000017"); }
fn raders_000151(c: &mut Criterion) { bench_raders(c,   151, "raders_000151"); }
fn raders_000257(c: &mut Criterion) { bench_raders(c,   257, "raders_000257"); }
fn raders_001009(c: &mut Criterion) { bench_raders(c,  1009, "raders_001009"); }
fn raders_002017(c: &mut Criterion) { bench_raders(c,  2017, "raders_002017"); }
fn raders_065537(c: &mut Criterion) { bench_raders(c, 65537, "raders_065537"); }
fn raders_746497(c: &mut Criterion) { bench_raders(c,746497, "raders_746497"); }

criterion_group!(raders, raders_000005, raders_000017, raders_000151, raders_000257,
                         raders_001009, raders_002017, raders_065537, raders_746497);

/// Times just the FFT setup (not execution)
/// for a given length, specific to Rader's algorithm
fn bench_raders_setup(c: &mut Criterion, len: usize, id: &str) {

    let mut planner = rustfft::FFTplanner::new(false);
    let inner_fft = planner.plan_fft(len - 1);

    c.bench_function(id, |b| b.iter(|| { 
        let fft : Arc<dyn FFT<f32>> = Arc::new(RadersAlgorithm::new(len, Arc::clone(&inner_fft)));
        black_box(fft);
    }));
}

fn raders_setup_000005(c: &mut Criterion) { bench_raders_setup(c,      5, "raders_setup_000005"); }
fn raders_setup_000017(c: &mut Criterion) { bench_raders_setup(c,     17, "raders_setup_000017"); }
fn raders_setup_000151(c: &mut Criterion) { bench_raders_setup(c,    151, "raders_setup_000151"); }
fn raders_setup_000257(c: &mut Criterion) { bench_raders_setup(c,    257, "raders_setup_000257"); }
fn raders_setup_001009(c: &mut Criterion) { bench_raders_setup(c,   1009, "raders_setup_001009"); }
fn raders_setup_002017(c: &mut Criterion) { bench_raders_setup(c,   2017, "raders_setup_002017"); }
fn raders_setup_065537(c: &mut Criterion) { bench_raders_setup(c,  65537, "raders_setup_065537"); }
fn raders_setup_746497(c: &mut Criterion) { bench_raders_setup(c, 746497, "raders_setup_746497"); }

criterion_group!(raders_setup, raders_setup_000005, raders_setup_000017, raders_setup_000151, raders_setup_000257,
                               raders_setup_001009, raders_setup_002017, raders_setup_065537, raders_setup_746497);

criterion_main!(complex_p2, complex_p7, complex_prime, complex_primepower, complex_composite,
                good_thomas, good_thomas_setup, good_thomas_butterfly,
                mixed_radix, mixed_radix_butterfly,
                raders, raders_setup);