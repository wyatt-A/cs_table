use std::io::Write;
use std::fs::File;
use array_lib::ArrayDim;
use array_lib::io_nifti::write_nifti;
use rand::{rng, RngExt};
use rayon::prelude::*;

fn main() {

    let nx = 3000;
    let ny = 1562;

    let accel = 8.;

    let target_samples = (nx as f64 * ny as f64 / accel).ceil() as usize;

    //let pa = 1.8;
    let pa = 1.8;
    //let pb = 5.4;
    let pb = 9.55;
    let prob_offset = 0.05;

    let pdf_x = |x: f64| (-(pb * x / nx as f64).powf(pa)).exp().sqrt();
    let pdf_y = |y: f64| (-(pb * y / ny as f64).powf(pa)).exp().sqrt();

    let dims = ArrayDim::from_shape(&[nx, ny]);

    let mut pdf = dims.alloc(0f64);
    pdf.par_iter_mut().enumerate().for_each(|(i, x)| {
        let [ix,iy,..] = dims.calc_idx_centered(i);
        *x = (pdf_x(ix.abs() as f64) * pdf_y(iy.abs() as f64) + prob_offset).min(1.0);
    });

    let s = pdf.iter().sum::<f64>();

    let mut rng = rng();
    rng.random_range(0.0..1.0);

    let mut points = vec![];
    let mut count = 0;
    let mut samples:Vec<f32> = pdf.iter().enumerate().map(|(i,p)|{
        if rng.random_range(0.0..1.0) < *p {
            let [ix,iy,..] = dims.calc_idx_centered(i);
            points.push(ix);
            points.push(iy);
            count += 1;
            1.
        }else {
            0.
        }
    }).collect();

    println!("s = {}, target = {}",s,target_samples);
    println!("c = {}",count);

    let mut tmp = vec![0.;samples.len()];
    dims.fftshift(&samples,&mut tmp, true);
    samples.copy_from_slice(&tmp);

    let pairs:Vec<_> = points.chunks_exact(2).collect();

    let r:Vec<_> = pairs.iter().map(|pair|{
        (pair[0] * pair[0] + pair[1] * pair[1]) as f64
    }).collect();

    let mut r_idx = (0..r.len()).collect::<Vec<_>>();
    r_idx.sort_by(|&a, &b| {
        r[a].total_cmp(&r[b])
    });

    let sorted_points:Vec<_> = r_idx.iter().map(|&i|[pairs[i][0],pairs[i][1]]).flatten().collect();

    let mut f = File::create("bf_cs_table.txt").unwrap();
    for point in sorted_points {
        writeln!(&mut f, "{}", point).unwrap();
    }

    write_nifti("test",&samples,dims);

}