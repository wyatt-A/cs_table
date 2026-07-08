use std::path::PathBuf;
use array_lib::ArrayDim;
use array_lib::io_cfl::write_cfl;
use clap::Parser;
use num_complex::Complex32;
use cs_table::vdpd::generate_vdpd_mask;
use rayon::prelude::*;

#[derive(Parser)]
struct Args {

    /// output mask as cfl array
    output_cfl: PathBuf,

    /// grid size along y-axis
    #[arg(long,default_value_t = 256)]
    ny:usize,

    /// grid size along z-axis
    #[arg(long,default_value_t = 256)]
    nz:usize,

    /// number of independently sampled masks to generate
    #[arg(long,short,default_value_t = 1)]
    n_masks:usize,

    /// parameter that controls sample density
    #[arg(long,short,default_value_t = 15.)]
    gamma:f32,

    /// normalized full-samples center radius
    #[arg(long,short,default_value_t = 0.05)]
    cr:f32,

    /// small value preventing a zero-radius at center
    #[arg(long,short,default_value_t = 0.015)]
    delta:f32,

    /// estimate gamma for a given acceleration factor
    #[arg(long,short)]
    estimate_gamma: Option<f32>,
}



fn main() {

    let args = Args::parse();

    let mask_data:Vec<_> = (0..args.n_masks).into_par_iter().map(|i| {
        println!("generating mask {i}");
        let mask = generate_vdpd_mask(
            args.ny,
            args.nz,
            1.0,   // acceleration
            args.cr,  // fully sampled center radius in normalized k-space
            args.gamma,  // gamma: larger gives denser sampling
            args.delta, // delta: prevents radius from going to zero at center
            i as u64,
        );
        mask
    }).collect();

    let n = mask_data.iter().flatten().filter(|x|**x).count();
    let acc = (args.ny * args.nz * args.n_masks) as f32 / n as f32;
    println!("acceleration: {}",acc);
    let mask_data:Vec<Complex32> = mask_data.into_iter().flatten().map(|x| if x { Complex32::ONE } else { Complex32::ZERO }).collect();
    write_cfl(args.output_cfl,&mask_data,ArrayDim::from_shape(&[args.ny,args.nz,args.n_masks]));

}