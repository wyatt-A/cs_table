use civm_rust_utils::write_to_file;
use itertools::Itertools;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::fmt::Display;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub const MAX_TABLE_ELEMENTS: usize = 196095;
const RNG_SEED: u64 = 42; // seed for pseudo-random shuffling

pub struct CSTable {
    pub source: PathBuf,
    pub elements: Vec<i16>,
}
pub struct KspaceCoord {
    pub k_phase: i16,
    pub k_slice: i16,
}

impl CSTable {
    pub fn open<T: AsRef<Path>>(source: T) -> Self {
        if !source.as_ref().exists() {
            println!("{:?}", source.as_ref());
            panic!("cs table not found!");
        }
        let mut s = String::new();
        let mut f = File::open(&source).expect("cannot open file");
        f.read_to_string(&mut s).expect("cannot read from file");
        // we are expecting a list of integers
        let e = s.lines().flat_map(|line| line.parse()).collect();
        Self {
            source: source.as_ref().to_owned(),
            elements: e,
        }
    }

    pub fn from_i16_pairs(file_to_write: &Path, pairs: &[[i16; 2]]) -> Self {
        let elements = flatten(pairs.to_owned());
        let s = Self {
            source: file_to_write.to_path_buf(),
            elements,
        };
        s.write();
        s
    }

    pub fn from_i32_pairs(file_to_write: &Path, pairs: &[[i32; 2]]) -> Self {
        let mut elements = Vec::<i16>::with_capacity(pairs.len());
        pairs.iter().for_each(|pair| {
            if pair[0] > i16::MAX as i32
                || pair[0] < i16::MIN as i32
                || pair[1] > i16::MAX as i32
                || pair[1] < i16::MIN as i32
            {
                panic!("i16 overflow from i32 cs table pairs! {:?}", pair);
            }
            elements.push(pair[0] as i16);
            elements.push(pair[1] as i16);
        });
        let s = Self {
            source: file_to_write.to_path_buf(),
            elements,
        };
        s.write();
        s
    }

    pub fn write(&self) {
        let mut s = String::new();
        for entry in &self.elements {
            s.push_str(&entry.to_string());
            s.push('\n');
        }
        write_to_file(&self.source, None, &s);
    }

    pub fn n_elements(&self) -> usize {
        self.elements.len()
    }

    pub fn elements(&self) -> Vec<i16> {
        self.elements.clone()
    }

    pub fn n_views(&self) -> usize {
        self.elements.len() / 2
    }

    pub fn coordinates(&self, read_element_offset: usize) -> Vec<KspaceCoord> {
        if (self.n_elements() % 2) != 0 {
            panic!("table must have an even number of elements");
        }
        let mut coords = Vec::<KspaceCoord>::with_capacity(self.n_elements() / 2);
        let range = read_element_offset..self.n_elements() / 2;
        for i in range {
            coords.push(KspaceCoord {
                k_phase: self.elements[2 * i],
                k_slice: self.elements[2 * i + 1],
            })
        }
        coords
    }

    pub fn coordinate_pairs(&self, read_element_offset: usize, index_offset: i32) -> Vec<[i32; 2]> {
        if (self.n_elements() % 2) != 0 {
            panic!("table must have an even number of elements");
        }
        let mut coords = Vec::<[i32; 2]>::with_capacity(self.n_elements() / 2);
        let range = read_element_offset..self.n_elements() / 2;
        for i in range {
            coords.push([
                self.elements[2 * i] as i32 + index_offset,
                self.elements[2 * i + 1] as i32 + index_offset,
            ])
        }
        coords
    }

    pub fn coordinate_pairs_i16(&self) -> Vec<[i16; 2]> {
        self.coordinate_pairs(0, 0)
            .iter()
            .map(|coord| [coord[0] as i16, coord[1] as i16])
            .collect()
    }

    pub fn indices(&self, read_element_offset: usize, matrix_size: [i16; 2]) -> Vec<(i16, i16)> {
        let phase_off = matrix_size[0] / 2;
        let slice_off = matrix_size[1] / 2;
        self.coordinates(read_element_offset)
            .iter()
            .map(|coord| (coord.k_phase + phase_off, coord.k_slice + slice_off))
            .collect()
    }

    pub fn copy_to(&self, dest: &Path, file_name: &str) {
        let mut s = String::new();
        let mut f = File::open(&self.source).expect("cannot open file");
        let fname = dest.join(file_name);
        f.read_to_string(&mut s).expect("cannot read from file");
        let mut d = File::create(fname).expect("cannot create file");
        d.write_all(s.as_bytes()).expect("trouble writing to file");
    }
}

// sets number of repetitions accordingly
pub trait CompressedSensing {
    fn set_cs_table(&mut self);
    fn cs_table(&self) -> PathBuf;
}

pub fn make_fse_table<T: AsRef<Path>>(
    filepath: T,
    coordinates: &[[i32; 2]],
    view_accel: usize,
    fs_size: usize,
    dummy_excitations: usize,
) -> CSTable {
    // table is read in order of echoes, low freq to high freq

    let n_zeros = dummy_excitations * view_accel;

    let mut final_table = vec![[0i32; 2]; n_zeros];

    let mut fs_samples = vec![];
    // region where each echo gets same spatial encoding
    for i in 0..fs_size {
        for j in 0..fs_size {
            for _ in 0..view_accel {
                let p1 = i as i32 - fs_size as i32 / 2;
                let p2 = j as i32 - fs_size as i32 / 2;
                fs_samples.push([p1, p2]);
            }
        }
    }

    // sort coordinates based on spatial frequency low to high
    let coords = coordinates.to_owned();
    let rsq: Vec<_> = coords
        .iter()
        .map(|pair| (pair[0] as i32).pow(2) + (pair[1] as i32).pow(2))
        .collect();
    let mut indices: Vec<_> = (0..coords.len()).collect();
    indices.sort_by_key(|&i| rsq[i]);
    let mut sorted_coords: Vec<_> = indices.iter().map(|&i| coords[i]).collect();

    // interleave low and high freqs
    let mut interleaved = vec![];
    let s_len = sorted_coords.len();

    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    sorted_coords[0..s_len / 2].shuffle(&mut rng);
    sorted_coords[s_len / 2..].shuffle(&mut rng);

    sorted_coords[0..s_len / 2]
        .iter()
        .zip(sorted_coords[s_len / 2..].iter())
        .for_each(|(lf, hf)| {
            interleaved.push(*lf);
            interleaved.push(*hf);
        });

    final_table.append(&mut fs_samples);
    final_table.append(&mut interleaved);

    CSTable::from_i32_pairs(filepath.as_ref(), &final_table)
}

fn flatten(coords: Vec<[i16; 2]>) -> Vec<i16> {
    let mut interleaved = Vec::<i16>::with_capacity(coords.len() * 2);
    coords.into_iter().for_each(|pair| {
        interleaved.push(pair[0]);
        interleaved.push(pair[1]);
    });
    interleaved
}
