use std::f32::consts::PI;
use array_lib::ArrayDim;
use array_lib::io_cfl::write_cfl;
use array_lib::io_nifti::write_nifti;
use num_complex::Complex;
use rand::Rng;
use cs_table::bridson::{poisson_disc_bridson_2d, poisson_disc_bridson_3d, Point2};

//r = linspace(0,1,1000);
//
// rc = 0.1;
// rt = 0.8;
// b  = 0.1;
//
// prob = zeros(size(r));
//
// for i = 1:length(r)
//     ri = r(i);
//
//     if ri <= rc
//         p = 1;
//     elseif ri <= rt
//         t = (ri - rc) / (rt - rc);   % normalize 0→1
//
//         % Quintic smoothstep (C2 continuous)
//         s = t^3 * (t * (6*t - 15) + 10);
//
//         % invert so it goes from 1 → b
//         p = (1 - b) * (1 - s) + b;
//     else
//         p = b;
//     end
//
//     prob(i) = p;
// end

fn main() {


    let mut rng = rand::rng();
    let points = poisson_disc_bridson_3d(1.,1.,1.,0.03,30,&mut rng);


    let coords_dims = ArrayDim::from_shape(&[3,points.len()]);
    let mut coords = vec![];

    for point in points {
        coords.push(Complex::new(point.x,0.));
        coords.push(Complex::new(point.y,0.));
        coords.push(Complex::new(point.z,0.));
    }

    write_cfl("scatter3",&coords,coords_dims);

}



// fn main() {
//
//
//     let width = 2.;
//     let height = 2.;
//     let r = 0.003;
//     let k = 30;
//
//     let mut rng = rand::rng();
//
//     let points = poisson_disc_bridson_2d(width,height,r,k,&mut rng);
//
//
//     let mut scatter = vec![];
//
//     for point in points {
//
//         let x = point.x - 1.;
//         let y = point.y - 1.;
//
//         let r = (x * x + y * y).sqrt();
//         if r < 1. {
//             let [x,y] = poisson_to_generalized_gaussian_radial(x, y, 0.35, 0.5);
//             scatter.push(Complex::new(x, y));
//         }
//
//     }
//
//     println!("num points: {}", scatter.len());
//
//     write_cfl("scatter",&scatter,ArrayDim::from_shape(&[scatter.len()]));
//
// }


pub fn poisson_to_gaussian_radial(x: f32, y: f32, sigma: f32) -> [f32; 2] {
    let r = (x * x + y * y).sqrt();

    if r < 1e-8 {
        return [0.0, 0.0];
    }

    // assumes r is in [0, 1)
    let u = (r * r).clamp(0.0, 1.0 - 1e-8);

    let rp = sigma * (-2.0 * (1.0 - u).ln()).sqrt();

    let s = rp / r;
    [x * s, y * s]
}

pub fn poisson_to_generalized_gaussian_radial(
    x: f32,
    y: f32,
    alpha: f32,
    beta: f32,
) -> [f32; 2] {
    assert!(alpha > 0.0);
    assert!(beta > 0.0);

    let r = (x * x + y * y).sqrt();

    if r < 1e-8 {
        return [0.0, 0.0];
    }

    // Assumes input points are uniform in unit disk.
    let u = (r * r).clamp(0.0, 1.0 - 1e-8);

    // Generalized Gaussian / stretched exponential radial inverse CDF.
    let rp = alpha * (-(1.0 - u).ln()).powf(1.0 / beta);

    let scale = rp / r;

    [x * scale, y * scale]
}

pub fn poisson_to_radial_icdf<F>(
    x: f32,
    y: f32,
    inv_cdf: F,
) -> [f32; 2]
where
    F: Fn(f32) -> f32,
{
    let r = (x * x + y * y).sqrt();

    if r < 1e-8 {
        return [0.0, 0.0];
    }

    // For points uniform in the unit disk:
    // P(R <= r) = r^2
    let u = (r * r).clamp(0.0, 1.0 - 1e-8);

    let rp = inv_cdf(u);
    assert!(rp.is_finite());
    assert!(rp >= 0.0);

    let scale = rp / r;

    [x * scale, y * scale]
}