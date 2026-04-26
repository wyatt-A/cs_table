use std::f32::consts::PI;
use array_lib::ArrayDim;
use array_lib::io_cfl::write_cfl;
use array_lib::io_nifti::write_nifti;
use num_complex::{Complex, ComplexFloat};
use rand::Rng;
use cs_table::bridson::{poisson_disc_bridson_2d, poisson_disc_bridson_3d, Point2};
use cs_table::utils::{cumtrapz, interp1, trapz};

fn main() {


    let mut rng = rand::rng();
    let points = poisson_disc_bridson_3d(1.,1.,1.,0.01,30,&mut rng);

    let mut x = vec![];
    let mut y = vec![];
    let mut z = vec![];
    for point in points {

        let xi = point.x * 2. - 1.;
        let yi = point.y * 2. - 1.;
        let zi = point.z * 2. - 1.;

        let r = (xi.powi(2) + yi.powi(2)).sqrt();
        if r < 1. {
            x.push(xi);
            y.push(yi);
            z.push(zi);
        }

    }

    // background
    let b = 0.2;
    // transition
    let rt = 0.9;
    // center rad
    let rc = 0.9;

    let n_cdf_samples = 1000;

    let mut r = vec![];
    let mut p = vec![];

    (0..n_cdf_samples).for_each(|i| {
        let ri = i as f32 / (n_cdf_samples as f32 - 1.);
        r.push(ri);
        p.push(radial_profile(ri,rc,rt,b))
    });

    // normalize probability
    let integral = trapz(&r,&p);
    p.iter_mut().for_each(|p| *p /= integral);

    let mut cdf = vec![0.;n_cdf_samples];
    cumtrapz(&r,&p,&mut cdf);

    let mut icdf = vec![0.;x.len()];

    let u = x.iter().zip(y.iter()).map(|(x,y)|{
        x*x + y*y
    }).collect::<Vec<_>>();

    interp1(&cdf,&r,&u,&mut icdf);

    // move the points
    x.iter_mut().zip(y.iter_mut()).enumerate().for_each(|(i,(x,y))|{
        let scale = icdf[i] / (x.powi(2) + y.powi(2)).sqrt();
        *x *= scale;
        *y *= scale;
    });


    let scale = 128;
    // scale by grid size
    for i in 0..x.len() {
        x[i] = (scale as f32 * x[i]).round();
        y[i] = (scale as f32 * y[i]).round();
        z[i] = (scale as f32 * z[i]).round();
    }

    let grid_dims = ArrayDim::from_shape(&[2 * scale,2 * scale,2 * scale]);
    let mut grid = grid_dims.alloc(0.);

    for i in 0..x.len() {
        let idx = [
            x[i].round() as isize,
            y[i].round() as isize,
            z[i].round() as isize,
        ];
        let addr = grid_dims.calc_addr_signed(&idx);
        grid[addr] = 1.;
    }

    let mut shifted = grid_dims.alloc(0.);
    grid_dims.fftshift(&grid,&mut shifted,true);

    write_nifti("samples",&shifted,grid_dims);

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

pub fn radial_profile(r: f32, rc: f32, rt: f32, b: f32) -> f32 {
    assert!(rc >= 0.0 && rc < rt, "require 0 <= rc < rt");
    assert!(rt <= 1.0, "rt should be <= 1 (assuming normalized radius)");
    assert!(b >= 0.0 && b <= 1.0, "b should be in [0,1]");

    // Optional: clamp r to [0,1] if that's your domain
    let r = r.clamp(0.0, 1.0);

    if r <= rc {
        1.0
    } else if r <= rt {
        let t = (r - rc) / (rt - rc); // normalize 0→1

        // Quintic smoothstep (C2 continuous)
        let s = t * t * t * (t * (6.0 * t - 15.0) + 10.0);

        // Map from 1 → b
        (1.0 - b) * (1.0 - s) + b
    } else {
        b
    }
}