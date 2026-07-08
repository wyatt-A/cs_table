// let raw_pdf = gauss_pdf2(nx, ny, pa, pb);
// let (scaled_pdf, target_samples) = scale_pdf(raw_pdf, undersample_frac).unwrap();

use array_lib::ArrayDim;
use array_lib::io_nifti::write_nifti;
use rand::RngExt;

#[cfg(test)]
mod tests {
    use array_lib::ArrayDim;
    use array_lib::io_nifti::write_nifti;
    use crate::vd::{gen_pdf, gen_vd_mask};
    use rayon::prelude::*;

    #[test]
    fn vd() {

        let nx = 480;
        let ny = 480;
        let pa = 1.8;
        let pb = 5.4;

        let frac = 1./8.;

        let n_masks = 61;

        let mask_dims = ArrayDim::from_shape(&[nx,ny,n_masks]);
        let mut mask_buff = mask_dims.alloc(0f32);

        let pdf = gen_pdf(nx, ny, pa, pb, frac);

        mask_buff.par_chunks_exact_mut(nx*ny).for_each(|mask|{
            let m:Vec<f32> = gen_vd_mask(&pdf).into_iter().map(|x| if x {1.} else {0.0}).collect();
            mask.copy_from_slice(&m);
        });


        write_nifti("out",&mask_buff,ArrayDim::from_shape(&[nx,ny,n_masks]));


    }
}

pub fn gen_vd_mask(pdf: &[f64]) -> Vec<bool> {
    let n = pdf.iter().sum::<f64>().round() as usize;
    let mut rng = rand::rng();
    let mut result = vec![false;pdf.len()];
    let mut z = 0;

    let max_tries = 10_000;
    let mut count = 0;

    while z != n {
        result.iter_mut().zip(pdf).for_each(|(x,p)|{
            *x = rng.random::<f64>() < *p
        });
        z = result.iter().filter(|x|**x).count();
        count += 1;
        if count >= max_tries {
            panic!("failed to generate mask within 10_000 tries");
        }
    }
    result
}

pub fn gen_pdf(nx: usize, ny: usize, pa: f64, pb: f64, frac: f64) -> Vec<f64> {
    let (updf,..) = pdf(nx,ny,pa,pb);
    let (pdf,..) = scale_pdf(&updf,nx,ny,frac);
    pdf
}

/// generates an un-normalized 2D bell curved with shape parameters pa and pb on a grid
/// of size nx by ny
fn pdf(nx: usize, ny: usize, pa: f64, pb: f64) -> (Vec<f64>, ArrayDim) {
    let pdf_x = |x: f64| (-(pb * x / nx as f64).powf(pa)).exp().sqrt();
    let pdf_y = |y: f64| (-(pb * y / ny as f64).powf(pa)).exp().sqrt();

    let pdf_dims = ArrayDim::from_shape(&[nx,ny]);
    let mut pdf = pdf_dims.alloc(0f64);

    pdf.iter_mut().enumerate().for_each(|(addr, p)| {
        let [x,y,..] = pdf_dims.calc_idx_centered(addr);
        let r = ((x.pow(2) + y.pow(2)) as f64).sqrt();
        *p = pdf_x(r) * pdf_y(r);
    });

    let mut dst = pdf_dims.alloc(0f64);
    pdf_dims.fftshift(&pdf,&mut dst,true);
    pdf.copy_from_slice(&dst);

    (pdf,pdf_dims)

}

fn scale_pdf(
    pdf: &[f64],
    nx:usize,
    ny:usize,
    undersample_frac: f64,
) -> (Vec<f64>,usize) {

    assert!(undersample_frac < 1., "undersampling must be less than 1");

    let grid_size = nx * ny;
    assert_eq!(pdf.len(),grid_size);

    // target samples represents the target energy of the pdf
    let target_samples = (grid_size as f64 * undersample_frac).round();

    //println!("target samples = {}", target_samples);
    // the goal is to have the sum of the pdf be equal to target_samples
    let mut s = pdf.iter().sum::<f64>();

    assert!(target_samples > s, "target samples too low");

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
            panic!("max iterations of {} exceeded",max_iter);
        }

        offset_mid = (offset_lb + offset_hb) / 2.0;

        // set values above 1 to 1
        s = pdf.iter().map(|x| *x + offset_mid).map(|x|x.min(1.)).sum();

        if s < target_samples {
            offset_lb = offset_mid;
        } else {
            offset_hb = offset_mid;
        }
        iter_count += 1;
    }

    let scaled_pdf:Vec<f64> = pdf.iter().map(|x|{
        (*x + offset_mid).min(1.)
    }).collect();

    (scaled_pdf, target_samples as usize)
}
