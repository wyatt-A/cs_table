use std::path::PathBuf;

use clap::Parser;
use cs_table::sampling::downsample_view_table;
use cs_table::sampling::gen_sampling;
use cs_table::sampling::grid2;
use cs_table::ViewTable;


#[derive(clap::Parser, Debug)]
pub struct Args {
    pub output_file:PathBuf,
    /// size of cs-table
    pub x:usize,

    /// max number of iterations for monte-carlo solver
    #[clap(long,short)]
    pub iterations:Option<usize>,

    /// undersampling fraction
    #[clap(long,short)]
    pub sample_frac:Option<f64>,

    /// acceleration rate
    #[clap(long)]
    pub accel:Option<f64>,

    #[clap(long)]
    /// optional y-dimension
    pub y:Option<usize>,
    #[clap(long,short='a')]
    /// pdf shape parameter a
    pub pa:Option<f64>,
    #[clap(long,short='b')]
    /// pdf shape parameter b
    pub pb:Option<f64>,

    #[clap(long)]
    /// sample deviation tolerance
    pub tol:Option<usize>,

    #[clap(long,short)]
    /// sample deviation tolerance
    pub target_scanner:Option<String>,

    /// generate a cs table that is a sub-set of a parent table
    #[clap(long)]
    pub parent_table:Option<PathBuf>

}

fn main() {
    
    let args = Args::parse();

    let pa = args.pa.unwrap_or(1.8);
    let pb = args.pb.unwrap_or(5.4);
    let sample_frac = args.sample_frac.unwrap_or_else(||{
        1./args.accel.unwrap_or(8.)
    });

    let tol = args.tol.unwrap_or(0);
    let nx = args.x;
    let ny = args.y.unwrap_or(nx);
    let iter = args.iterations.unwrap_or(100);

    let result = if let Some(parent_table) = args.parent_table.as_ref() {

        let p = ViewTable::from_file(parent_table)
        .expect("failed to load parent view table");

        println!("downsampling cs table with paramters:
        nx:             {}
        ny:             {}
        undersampling:  {}
        pa:             {}
        pa:             {}
        ",
            nx,ny,sample_frac,pa,pb);

        downsample_view_table(&p, nx, ny, pa, pb, sample_frac)
            .expect("failed to find table with current settings")

    } else {
        println!("running cs-table-gen with paramters:
        nx:             {}
        ny:             {}
        undersampling:  {}
        pa:             {}
        pa:             {}
        tolerance:      {}
        iterations:     {}
        ",
            nx,ny,sample_frac,pa,pb,tol,iter);
        
            let mask = gen_sampling(nx, ny, pa, pb, sample_frac, tol, iter);
            let (x, y) = grid2::<i32>(nx, ny);
        
            let mut coords = vec![];
        
            mask.into_iter()
                .zip(x.into_iter().zip(y.into_iter()))
                .for_each(|(m, (x, y))| {
                    if m {
                        coords.push([y, x]);
                    }
                });
        
            ViewTable::from_coord_pairs(&coords).unwrap()
    };

    if let Some(scanner) = &args.target_scanner {
        let header = match scanner.to_ascii_lowercase().as_str() {
            "b" | "bruker" => {
                format!("dim_x={}\ndim_y={}\nn_coords={}\n",
                nx,ny,result.n_coord_pairs()
            )
            },
            _=> String::new()
        };
        result.write_with_header(&args.output_file,header).expect("failed to write view table");
        return
    }
    result.write(&args.output_file).expect("failed to write view table");
}