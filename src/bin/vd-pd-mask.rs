use array_lib::ArrayDim;
use array_lib::io_nifti::write_nifti;
use cs_table::vdpd::generate_vdpd_mask;
use rayon::prelude::*;

fn main() {
    let ny = 256;
    let nz = 256;

    let n_masks = 61;

    let mask_data:Vec<_> = (0..n_masks).into_par_iter().map(|i| {
        println!("generating mask {i}");
        let mask = generate_vdpd_mask(
            ny,
            nz,
            8.0,   // acceleration
            0.05,  // fully sampled center radius in normalized k-space
            15.0,  // gamma: larger gives denser sampling
            0.015, // delta: prevents radius from going to zero at center
            i as u64,
        );
        mask
    }).collect();

    let mask_data:Vec<f32> = mask_data.into_iter().flatten().map(|x| if x { 1. } else { 0.0 }).collect();

    write_nifti("mask",&mask_data,ArrayDim::from_shape(&[256,256,n_masks]));

}