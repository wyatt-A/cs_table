use ndarray::{Array2, ShapeBuilder};
use num_complex::Complex32;
use num_traits::{One, PrimInt, Signed, ToPrimitive, Zero};
use sampling::grid2;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet, error::Error, fs::File, io::{Read, Write}, path::Path
};
use array_lib::ArrayDim;
use array_lib::io_cfl::write_cfl;

pub mod cs_table;
pub mod sampling;

/// type representing a lookup table for compressed phase encoding. By default, this is a list of 2-D
/// coordinates specifying a cartesian phase encoding. For 2-D phase encoding, the second coordinate is 0.
/// The file representation is a stream of integers seperated by newlines or any ascii whitespace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewTable {
    elements: Vec<i32>,
}



impl ViewTable {

    pub fn to_bytes(self) -> Vec<u8> {
        bincode::serialize(&self).expect("failed to serialize view table")
    }

    pub fn from_bytes(bytes:&[u8]) -> Self {
        bincode::deserialize(bytes).expect("failed to deserialize bytes to view table")
    }

    // return a boolean mask of the view table of a given size
    pub fn to_mask(self,nx:usize,ny:usize) -> Array2<bool> {

        let mut a = Array2::from_elem((nx,ny).f(), false);

        let coord_set = HashSet::<[i32;2]>::from_iter(
            self.coordinate_pairs::<i32>().unwrap().into_iter()
        );

        let (grid_x,grid_y) = grid2::<i32>(nx, ny);

        a.iter_mut().zip(grid_x.iter().zip(grid_y.iter())).for_each(|(a,(x,y))|{

            if coord_set.contains(&[*x,*y]) {
                *a = true;
            }

        });
        
        a

    }

    /// save view table as a 2D complex32 mask
    pub fn write_as_cfl(&self,nx:usize,ny:usize,filename:impl AsRef<Path>) {

        let dims = ArrayDim::from_shape(&[nx,ny]);

        let complex_mask = self.clone().to_mask(nx, ny).map(|x|{
            if *x {
                Complex32::one()
            }else {
                Complex32::zero()
            }
        }).into_dyn();
        write_cfl(&filename,complex_mask.as_slice_memory_order().unwrap(),dims)
    }

    // create a view table from a boolean mask
    pub fn from_mask(mask:Array2<bool>) -> Self {

        let shape = mask.shape();
        let nx = shape[0];
        let ny = shape[1];


        let mut coords = vec![];

        let (grid_x, grid_y) = grid2::<i32>(nx, ny);

        mask.into_iter().zip(grid_x.iter().zip(grid_y.iter())).for_each(|(m,(x,y))|{
            if m {
                coords.push([*x,*y]);
            }
        });
        
        Self::from_coord_pairs(&coords).unwrap()

    }


    /// instantiate view table from a file containing elements separated by unicode whitespace
    pub fn from_file<P: AsRef<Path>>(file: P) -> Result<Self, Box<dyn Error>> {
        Self::from_file_inner(file.as_ref(), true)
    }

    /// instantiate view table from a stream of data points
    pub fn from_stream<T: ToPrimitive + PrimInt + Signed>(
        stream: &[T],
    ) -> Result<Self, Box<dyn Error>> {
        if stream.len() % 2 != 0 {
            Err(format!(
                "expected an even number of elements, received {}",
                stream.len()
            ))?
        }
        let mut elements: Vec<i32> = Vec::<i32>::with_capacity(stream.len());
        for element in stream {
            elements.push(element.to_i32().ok_or("cannot convert to i32")?)
        }
        Ok(Self { elements })
    }

    /// writes to a file with a newline delimeter
    pub fn write<P: AsRef<Path>>(&self, file: P) -> Result<(), Box<dyn Error>> {
        self.write_with_delimeter(file, "\n")
    }

    /// writes to a file with a leading header
    pub fn write_with_header(&self, file: impl AsRef<Path>,header: impl AsRef<str>) -> Result<(), Box<dyn Error>> {
        let mut f = File::create(file)?;
        let mut h = header.as_ref().to_string();
        h.push_str(&self.to_string("\n"));
        f.write_all(h.as_bytes())?;
        Ok(())
    }

    /// writes to a file with a specified whitespace delimeter. An error is returned if the delimeter
    /// is not unicode whitespace
    pub fn write_with_delimeter<P: AsRef<Path>>(
        &self,
        file: P,
        delimeter: &str,
    ) -> Result<(), Box<dyn Error>> {
        if !delimeter.trim().is_empty() {
            Err(format!(
                "non-whitespace view table delimeter ({}) specified",
                delimeter
            ))?
        }
        let mut f = File::create(file)?;
        f.write_all(self.to_string(delimeter).as_bytes())?;
        Ok(())
    }

    pub fn trim_start(self, n_coords: usize) -> Self {
        let new_elements = self.elements[n_coords * 2..].to_owned();
        Self {
            elements: new_elements,
        }
    }

    pub fn coordinate_pairs<T: ToPrimitive + PrimInt + Signed>(
        &self,
    ) -> Result<Vec<[T; 2]>, Box<dyn Error>> {
        let mut pairs = Vec::<[T; 2]>::with_capacity(self.n_coord_pairs());
        for pair in self.elements.chunks_exact(2) {
            let x = T::from(pair[0]).ok_or("cannot cast from i32 to specified type")?;
            let y = T::from(pair[1]).ok_or("cannot cast from i32 to specified type")?;
            pairs.push([x, y])
        }
        Ok(pairs)
    }

    /// return coordinate pairs with some stride n and offset k. This is useful for advances phase encoding
    /// sequences such as FSE
    pub fn coordinate_pairs_strided<T: ToPrimitive + PrimInt + Signed>(
        &self,
        n: usize,
        k: usize,
    ) -> Result<Vec<[T; 2]>, Box<dyn Error>> {
        let pairs = self.coordinate_pairs()?;
        Self::stride_at_index(pairs, n, k)
    }

    /// returns vector of elements at an index k with given stride n
    fn stride_at_index<T: Copy>(x: Vec<T>, n: usize, k: usize) -> Result<Vec<T>, Box<dyn Error>> {
        let mut strided = vec![];
        for chunk in x.chunks_exact(n) {
            let c = chunk.get(k).ok_or(format!(
                "stride offset ({}) out of range for stride length {}",
                k, n
            ))?;
            strided.push(*c);
        }
        Ok(strided)
    }

    pub fn from_coord_pairs<T: ToPrimitive + PrimInt + Signed>(
        pairs: &[[T; 2]],
    ) -> Result<Self, Box<dyn Error>> {
        let mut elements: Vec<i32> = Vec::with_capacity(pairs.len() * 2);
        for pair in pairs {
            let [x, y] = pair;
            elements.push(x.to_i32().ok_or("cannot convert to i32")?);
            elements.push(y.to_i32().ok_or("cannot convert to i32")?);
        }
        Ok(Self { elements })
    }

    pub fn n_coord_pairs(&self) -> usize {
        self.elements.len() / 2
    }

    fn to_string(&self, delimeter: &str) -> String {
        let mut s: String = self
            .elements
            .iter()
            .map(|elem| elem.to_string())
            .collect::<Vec<_>>()
            .join(delimeter);
        s.push('\n');
        s
    }

    fn from_file_inner(file: &Path, strict: bool) -> Result<Self, Box<dyn Error>> {
        let mut f = File::open(file)?;
        let mut s = String::new();

        f.read_to_string(&mut s)?;
        let mut elements = vec![];

        if strict {
            for char in s.split_ascii_whitespace() {
                elements.push(char.parse::<i32>()?);
            }
        } else {
            elements = s
                .split_whitespace()
                .flat_map(|char| char.parse::<i32>().ok())
                .collect();
        }

        if elements.len() % 2 != 0 {
            Err(format!(
                "recieved an odd number of elements ({}) from {}.
            Expecting an even number of elements.",
                file.to_string_lossy().to_string(),
                elements.len()
            ))?
        }

        Ok(Self { elements })
    }
}
