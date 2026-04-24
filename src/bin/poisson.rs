use std::f32::consts::PI;
use array_lib::ArrayDim;
use array_lib::io_cfl::write_cfl;
use num_complex::Complex;
use rand::Rng;
use cs_table::bridson::{poisson_disc_bridson_2d, Point2};

fn main() {


    let width = 2.;
    let height = 2.;
    let r = 0.003;
    let k = 30;

    let mut rng = rand::rng();

    let points = poisson_disc_bridson_2d(width,height,r,k,&mut rng);


    let mut scatter = vec![];

    for point in points {

        let x = point.x - 1.;
        let y = point.y - 1.;

        let r = (x * x + y * y).sqrt();
        if r < 1. {
            let [x,y] = poisson_to_generalized_gaussian_radial(x, y, 0.35, 0.5);
            scatter.push(Complex::new(x, y));
        }

    }

    println!("num points: {}", scatter.len());

    write_cfl("scatter",&scatter,ArrayDim::from_shape(&[scatter.len()]));

}


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