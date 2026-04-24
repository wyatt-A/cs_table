use crate::ViewTable;
use indicatif::ProgressStyle;
use itertools::{iproduct, Itertools};
use ndarray::{Array, Array2, ShapeBuilder};
use num_complex::{Complex32, ComplexFloat};
use num_traits::{FloatConst, Zero};
use rand::{RngExt, SeedableRng};
use rayon::{prelude::*, vec};
use core::num;
use std::collections::HashSet;
use std::error::Error;
use std::ops::Range;
use dft_lib::common::{FftDirection, NormalizationType};
use dft_lib::rs_fft::rs_fftn;
use rand::distr::Uniform;

///CS table downsampling ... given a sampling pattern, find a lower-resolution sampling pattern
///that is a sub-set of the original. This does not evaluate peak interference when finding a
///solution!
pub fn downsample_view_table(base_table:&ViewTable,target_nx:usize,target_ny:usize,pa:f64,pb:f64,sample_rate:f64) -> Option<ViewTable> {
    
    let sample_mask = base_table.clone().to_mask(target_nx, target_ny);

    let target_samples = ((target_nx*target_ny) as f64 * sample_rate).round();

    if base_table.n_coord_pairs() < target_samples as usize {
        println!("the number of target samples {} cannot be greater than the number of reference samples {}",target_samples,base_table.n_coord_pairs());
        return None
    }

    let pdf = gauss_pdf2(target_nx,target_ny,pa,pb);

    let mut rng = rand::rng();
    let range = Uniform::new(0.0, 1.0);

    let mut undersample_factor_lb = 0.;
    let mut undersample_factor_ub = 1.;

    let max_loop_count = 1000;

    let mut loop_counter = 0;
    loop {

        if loop_counter == max_loop_count {
            return None
        }

        let mut mean_err = 0;
        let n = 100;
    
        let mid_comp_factor = (undersample_factor_lb + undersample_factor_ub)/2.;
        for _ in 0..n {
            let (scaled_pdf,_) = scale_pdf(pdf.clone(),mid_comp_factor).unwrap();
            let msk = scaled_pdf.map(|x| {
                if rng.random_range(0.0..1.0) < *x {
                    true
                } else {
                    false
                }
            });
            let t = &msk & &sample_mask;
            let total_samps = t.iter().fold(0, |acc, x| if *x {acc + 1} else {acc});
            let err = total_samps - target_samples as i32;

            if err == 0 {
                return Some(ViewTable::from_mask(t))
            }

            mean_err += err
        };
        mean_err /= n;
    
        
        if mean_err > 0 { // too many samples
            undersample_factor_ub = mid_comp_factor;
        }else { // too few samples
            undersample_factor_lb = mid_comp_factor;
        }
    
        println!("mean sample err = {}",mean_err);

        loop_counter += 1;
    }

}

/*
    PDF Generation for CS Sampling
*/
/// generates a mask of boolean values representing a CS phase encoding strategy. The mask is of size
/// nx-by-ny. The shape factors pa and pb determine the shape of the sampling PDF. The undersampling
/// fraction determines the sampling rate where 1 is fully-sampled. Tolerance determines the allowable
/// deviation from the number of samples determined by the undersampling fraction. The number of
/// iterations determines the number of simulations to be run. The more iterations, the better the
/// result at the cost of time.
pub fn gen_sampling(
    nx: usize,
    ny: usize,
    pa: f64,
    pb: f64,
    undersample_frac: f64,
    tolerance: usize,
    num_iter: usize,
) -> Array2<bool> {
    let raw_pdf = gauss_pdf2(nx, ny, pa, pb);
    let (scaled_pdf, target_samples) = scale_pdf(raw_pdf, undersample_frac).unwrap();

    // f32 version of scale for repeated calls
    let scale = scaled_pdf.map(|x| *x as f32);

    let mut min_sidelobe = f32::INFINITY;

    let mut rng = rand::rng();
    let range = Uniform::new(0.0, 1.0);

    let mut best_sidelobe = vec![];
    let mut best_mask = vec![];


    let prog_bar = indicatif::ProgressBar::new(num_iter as u64);
    prog_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}").unwrap()
            .progress_chars("#>-")
    );

    println!("running monte-carlo solver with {} iterations ...",num_iter);
    // main monte-carlo loop
    for _ in 0..num_iter {
        // random sampling / tolerance loop
        let msk = loop {
            let mut total_samps = 0;
            let msk = scaled_pdf.map(|x| {
                if rng.random_range(0.0..1.0) < *x {
                    total_samps += 1;
                    // we set complex values so we can perform fft on them later
                    Complex32::new(1., 0.)
                } else {
                    Complex32::new(0., 0.)
                }
            });
            // if total samples is within tolerance, return the mask to the for-loop
            if (total_samps - target_samples as i32).abs() as usize <= tolerance {
                break msk;
            }
        };

        // scale the mask such that less-probable samples have higher intensity
        let mut scaled_msk = (&msk / &scale).into_dyn();

        rs_fftn(scaled_msk.as_slice_memory_order_mut().unwrap(),&[nx,ny],FftDirection::Inverse,NormalizationType::Unitary);

        // retrieve the max side-lobe value
        let mut abs = scaled_msk.map(|x| x.abs());
        let f = abs.as_slice_memory_order_mut().unwrap();
        f[0] = 0.;
        let m = *f.iter().max_by(|a, b| a.partial_cmp(&b).unwrap()).unwrap();

        // check if result is better than the last
        if m < min_sidelobe {
            min_sidelobe = m;
            best_sidelobe.push(min_sidelobe);
            best_mask.push(msk);
        }

        prog_bar.inc(1);
    }
    prog_bar.finish_with_message("done");

    println!("side lobes: {:?}", best_sidelobe);
    let best = best_mask.pop().unwrap();
    // map from complex to bool
    best.map(|x| !x.is_zero())
}

#[derive(Debug)]
enum PdfScaleError {
    InvalidUnderSampleFrac(f64),
    TargetSamplesTooLow(f64, f64),
    MaxIterationsReached(usize),
}

fn scale_pdf(
    pdf: Array2<f64>,
    undersample_frac: f64,
) -> Result<(Array2<f64>, usize), PdfScaleError> {
    if undersample_frac > 1. {
        return Err(PdfScaleError::InvalidUnderSampleFrac(undersample_frac));
    }

    let grid_size: usize = pdf.shape().iter().product();

    // target samples represents the target energy of the pdf
    let target_samples = (grid_size as f64 * undersample_frac).round();

    //println!("target samples = {}", target_samples);
    // the goal is to have the sum of the pdf be equal to target_samples
    let mut s = pdf.sum();

    if s > target_samples as f64 {
        return Err(PdfScaleError::TargetSamplesTooLow(target_samples, s));
    }

    // a valid offset is bounded between 0 and 1
    let mut offset_lb: f64 = 0.;
    let mut offset_hb: f64 = 1.;
    let mut offset_mid = 0.;
    // Run bisection loop to find the correct offset
    let precision_limit = 1e-12;
    let max_iter = 20_000;
    let mut iter_count = 0;
    while (offset_hb - offset_lb).abs() >= precision_limit {
        if iter_count > max_iter {
            return Err(PdfScaleError::MaxIterationsReached(max_iter));
        }

        offset_mid = (offset_lb + offset_hb) / 2.0;

        // set values above 1 to 1
        s = (pdf.clone() + offset_mid).map(|x| x.min(1.)).sum();

        if s < target_samples {
            offset_lb = offset_mid;
        } else {
            offset_hb = offset_mid;
        }
        iter_count += 1;
    }

    let scaled_pdf = (pdf + offset_mid).map(|x| x.min(1.));
    let s = scaled_pdf.sum();

    // println!(
    //     "solution found in {} iterations with error {:0.2e} samples",
    //     iter_count,
    //     (s - target_samples).abs()
    // );
    Ok((scaled_pdf, target_samples as usize))
}

/// returns 2 2D grids varying over x or y
pub fn grid2<T: From<i32> + Copy>(nx: usize, ny: usize) -> (Array2<T>, Array2<T>) {
    let mut x_elems = Vec::with_capacity(nx * ny);
    let mut y_elems = x_elems.clone();

    for y in centered_range(ny) {
        for x in centered_range(nx) {
            x_elems.push(T::from(x));
            y_elems.push(T::from(y));
        }
    }

    (
        Array2::from_shape_vec((nx, ny).f(), x_elems).unwrap(),
        Array2::from_shape_vec((nx, ny).f(), y_elems).unwrap(),
    )
}

/// returns a range of length n centered about 0
fn centered_range(n: usize) -> Range<i32> {
    let n = n as i32;
    let half_n = n / 2;
    -half_n..half_n + n % 2
}

/// evaulates a 1D function over a 2D grid based on the l-2 norm of grid coordinates
fn pdf_eval<F>(pdf_1d: F, nx: usize, ny: usize) -> Array2<f64>
where
    F: Fn(f64) -> f64,
{
    let (x, y) = grid2::<f64>(nx, ny);
    let r = (x.map(|x| x.powi(2)) + y.map(|y| y.powi(2))).map(|r| r.sqrt());
    r.map(|r| pdf_1d(*r))
}

/// generates an un-normalized 2D gaussian curved with shape paramters pa and pb on a grid
/// of size nx by ny
fn gauss_pdf2(nx: usize, ny: usize, pa: f64, pb: f64) -> Array2<f64> {
    let pdf_x = |x: f64| (-(pb * x / nx as f64).powf(pa)).exp().sqrt();
    let pdf_y = |y: f64| (-(pb * y / ny as f64).powf(pa)).exp().sqrt();

    let (x, y) = grid2::<f64>(nx, ny);
    let r = (x.map(|x| x.powi(2)) + y.map(|y| y.powi(2))).map(|rsq| rsq.sqrt());

    r.map(|r| pdf_x(*r)) * r.map(|r| pdf_y(*r))
}

fn pdf(r: f32, nx: usize, ny: usize, a: f32, b: f32) -> f32 {
    let nx = nx as f32;
    let ny = ny as f32;

    let x = (-(b * r / nx).powf(a)).exp().sqrt();
    let y = (-(b * r / ny).powf(a)).exp().sqrt();

    x * y
}

fn grid_2d(x: Range<i32>, y: Range<i32>) -> Vec<[i32; 2]> {
    let mut out = Vec::with_capacity((x.len() * y.len()) as usize);
    for x_val in x {
        for y_val in y.clone() {
            out.push([x_val, y_val]);
        }
    }
    out
}

#[derive(Debug)]
pub struct Pdf {
    coordinates: Vec<[i32; 2]>,
    values: Vec<f32>,
    max_val: f32,
}

fn gen_pdf(nx: usize, ny: usize, a: f32, b: f32) -> Pdf {
    let hx = nx as i32 / 2;
    let hy = ny as i32 / 2;
    let ux = -hx + nx as i32;
    let uy = -hy + ny as i32;
    let coordinates = grid_2d(-hx..ux, -hy..uy);
    let mut max_val = 0.;
    let values = coordinates
        .iter()
        .map(|c| {
            let r = (c[0] * c[0] + c[1] * c[1]) as f32;
            let val = pdf(r, nx, ny, a, b);
            if val > max_val {
                max_val = val;
            }
            val
        })
        .collect();

    Pdf {
        coordinates,
        values,
        max_val,
    }
}

/// segment a view table into n_segments smaller view tables based on some distance function over
/// its coordinates
pub fn split_view_table<F>(
    view_tab: &ViewTable,
    n_segments: usize,
    distance_fn: F,
) -> Result<Vec<ViewTable>, Box<dyn Error>>
where
    F: Fn(&[i32; 2]) -> f64,
{
    let coords = view_tab.coordinate_pairs::<i32>()?;
    if coords.len() % n_segments != 0 {
        Err(format!(
            "number of coordinates ({}) not evenly divisible by number of segments ({})",
            coords.len(),
            n_segments
        ))?
    }
    let dist: Vec<f64> = coords.iter().map(|coord| distance_fn(coord)).collect();
    let mut pairs: Vec<_> = dist.iter().zip(coords.iter()).collect();
    pairs.sort_by(|a, b| {
        a.0.partial_cmp(b.0)
            .expect("cannot compare floating point values")
    });
    let sorted_coords: Vec<_> = pairs.into_iter().map(|(_, &b)| b).collect();
    let chunk_size = coords.len() / n_segments;
    let view_tabs: Vec<_> = sorted_coords
        .chunks_exact(chunk_size)
        .map(|chunk| ViewTable::from_coord_pairs(chunk).unwrap())
        .collect();
    Ok(view_tabs)
}

pub fn partition_view_table<F>(
    view_tab: &ViewTable,
    fractions: &[f32],
    distance_fn: F,
) -> Result<Vec<ViewTable>, Box<dyn Error>>
where
    F: Fn(&[i32; 2]) -> f64,
{
    let coords = view_tab.coordinate_pairs::<i32>()?;
    let dist: Vec<f64> = coords.iter().map(|coord| distance_fn(coord)).collect();
    let mut pairs: Vec<_> = dist.iter().zip(coords.iter()).collect();
    pairs.sort_by(|a, b| {
        a.0.partial_cmp(b.0)
            .expect("cannot compare floating point values")
    });
    let mut sorted_coords: Vec<_> = pairs.into_iter().map(|(_, &b)| b).collect();

    let t = fractions.iter().sum::<f32>();

    let mut f = fractions.to_vec();
    f.iter_mut().for_each(|x| *x /= t);

    let n = sorted_coords.len();

    // ensure that the distribution has the same number of coords
    let mut d: Vec<_> = f.into_iter().map(|x| (x * n as f32) as i32).collect();
    let diff = n as i32 - d.iter().sum::<i32>();
    *d.last_mut().unwrap() += diff;

    assert_eq!(d.iter().sum::<i32>() as usize, n);

    let mut view_tabs: Vec<_> = d
        .iter()
        .rev()
        .map(|&d| {
            let mut buff = vec![];
            for _ in 0..d as usize {
                buff.push(sorted_coords.pop().unwrap());
            }
            ViewTable::from_coord_pairs(&buff).unwrap()
        })
        .collect();
    view_tabs.reverse();
    Ok(view_tabs)
}

/// filter a view table based on some filter function, returning a new View table
pub fn filter_view_table<F>(view_tab: &ViewTable, filter_fn: F) -> Result<ViewTable, Box<dyn Error>>
where
    F: Fn(&[i32; 2]) -> bool,
{
    let coords = view_tab.coordinate_pairs::<i32>()?;
    let filtered: Vec<_> = coords.into_iter().filter(filter_fn).collect();
    ViewTable::from_coord_pairs(&filtered)
}

/// combine view tables into a single view table. If strict mode is enabled, every view will be unique
pub fn combine_view_tables(
    view_tables: &[ViewTable],
    strict: bool,
) -> Result<ViewTable, Box<dyn Error>> {
    let all_coords: Vec<_> = if strict {
        let mut h = HashSet::<[i32; 2]>::new();
        for view_tab in view_tables {
            for coord in view_tab.coordinate_pairs::<i32>()? {
                h.insert(coord);
            }
        }
        h.into_iter().collect()
    } else {
        let mut all_coords = vec![];
        for view_tab in view_tables {
            let mut x = view_tab.coordinate_pairs::<i32>()?;
            all_coords.append(&mut x);
        }
        all_coords
    };
    ViewTable::from_coord_pairs(&all_coords)
}

#[test]
fn test2() {
    let vt = ViewTable::from_file("../environment/cs_tables/stream_CS480_8x_pa18_pb54").unwrap();

    // let distance_fn = |coord: &[i32;2]| -> f64 {
    //     (coord[0].abs() + coord[1].abs()) as f64
    // };

    let distance_fn = |coord: &[i32; 2]| -> f64 {
        let c1 = coord[0] as f64 * f64::SQRT_2() / 2. - coord[1] as f64 * f64::SQRT_2() / 2.;
        let c2 = coord[0] as f64 * f64::SQRT_2() / 2. + coord[1] as f64 * f64::SQRT_2() / 2.;
        c1.abs() + c2.abs()
    };

    // let distance_fn = |coord: &[i32;2]| -> f64 {
    //     ((coord[0] as f64).powi(2) + (coord[1] as f64).powi(2)).sqrt()
    // };

    let view_tabs = split_view_table(&vt, 6, distance_fn).unwrap();

    for (i, view_tab) in view_tabs.iter().enumerate() {
        view_tab.write(format!("viewtab_{:02}", i)).unwrap()
    }
}
