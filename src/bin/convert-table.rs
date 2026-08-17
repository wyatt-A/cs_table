use std::path::PathBuf;
use array_lib::ArrayDim;
use array_lib::io_nifti::write_nifti;
use clap::Parser;
use cs_table::ViewTable;

#[derive(Parser,Debug)]
struct Args {
    input_txt: PathBuf,
    x: usize,
    y: usize,
    output_img: PathBuf,
}

fn main() {
    let args = Args::parse();
    let vt = ViewTable::from_file(args.input_txt).unwrap();
    let pairs = vt.coordinate_pairs::<isize>().unwrap();
    let dims = ArrayDim::from_shape(&[args.x, args.y]);
    let mut mask = dims.alloc(0f32);
    for pair in pairs.into_iter() {
        let addr = dims.calc_addr_signed(&[pair[1],pair[0]]);
        mask[addr] = 1.;
    }

    let mut tmp = dims.alloc(0f32);
    dims.fftshift(&mask, &mut tmp, true);
    mask.copy_from_slice(&tmp);

    write_nifti(args.output_img,&mask,dims);
}