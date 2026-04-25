use rand::{Rng, RngExt};
use std::f32::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2 {
    pub x: f32,
    pub y: f32,
}

/// Generate 2-D Poisson-disc samples using Bridson's algorithm.
///
/// # Arguments
/// - `width`: domain width
/// - `height`: domain height
/// - `r`: minimum allowed distance between samples
/// - `k`: number of candidates to try per active sample (commonly 30)
/// - `rng`: random number generator
///
/// # Returns
/// A vector of sample points in `[0,width) x [0,height)`.
pub fn poisson_disc_bridson_2d<R: Rng + ?Sized>(
    width: f32,
    height: f32,
    r: f32,
    k: usize,
    rng: &mut R,
) -> Vec<Point2> {
    assert!(width > 0.0, "width must be > 0");
    assert!(height > 0.0, "height must be > 0");
    assert!(r > 0.0, "r must be > 0");
    assert!(k > 0, "k must be > 0");

    let cell_size = r / 2.0_f32.sqrt();

    let grid_width = (width / cell_size).ceil() as usize;
    let grid_height = (height / cell_size).ceil() as usize;

    // Each grid cell stores at most one sample index.
    let mut grid: Vec<Option<usize>> = vec![None; grid_width * grid_height];

    let mut samples = Vec::<Point2>::new();
    let mut active = Vec::<usize>::new();

    let initial = Point2 {
        x: rng.random_range(0.0..=width),
        y: rng.random_range(0.0..=height),
    };

    samples.push(initial);
    active.push(0);

    let (gx, gy) = grid_coords(initial, cell_size);
    grid[grid_index(gx, gy, grid_width)] = Some(0);

    while !active.is_empty() {
        let active_i = rng.random_range(0..active.len());
        let sample_idx = active[active_i];
        let base = samples[sample_idx];

        let mut found = false;

        for _ in 0..k {
            let candidate = generate_annulus_candidate(base, r, rng);

            if candidate.x < 0.0
                || candidate.x >= width
                || candidate.y < 0.0
                || candidate.y >= height
            {
                continue;
            }

            if is_valid_candidate(candidate, &samples, &grid, grid_width, grid_height, cell_size, r)
            {
                let new_idx = samples.len();
                samples.push(candidate);
                active.push(new_idx);

                let (cgx, cgy) = grid_coords(candidate, cell_size);
                grid[grid_index(cgx, cgy, grid_width)] = Some(new_idx);

                found = true;
                break;
            }
        }

        if !found {
            active.swap_remove(active_i);
        }
    }

    samples
}

fn generate_annulus_candidate<R: Rng + ?Sized>(base: Point2, r: f32, rng: &mut R) -> Point2 {
    // Uniform angle
    let theta = rng.random_range(0.0..(2.0 * PI));

    // Radius in [r, 2r), sampled so area is uniform in the annulus.
    let u:f32 = rng.random_range(0.0..1.0);
    let radius = r * (1.0 + 3.0 * u).sqrt();
    // Explanation:
    // For annulus [r, 2r], area-uniform radius satisfies:
    // radius^2 ~ Uniform(r^2, (2r)^2) = Uniform(r^2, 4r^2)
    // so radius = sqrt(r^2 + u*(3r^2)) = r*sqrt(1+3u)

    Point2 {
        x: base.x + radius * theta.cos(),
        y: base.y + radius * theta.sin(),
    }
}

fn is_valid_candidate(
    p: Point2,
    samples: &[Point2],
    grid: &[Option<usize>],
    grid_width: usize,
    grid_height: usize,
    cell_size: f32,
    r: f32,
) -> bool {
    let (gx, gy) = grid_coords(p, cell_size);

    let x0 = gx.saturating_sub(2);
    let y0 = gy.saturating_sub(2);
    let x1 = (gx + 2).min(grid_width - 1);
    let y1 = (gy + 2).min(grid_height - 1);

    let r2 = r * r;

    for ny in y0..=y1 {
        for nx in x0..=x1 {
            if let Some(sample_idx) = grid[grid_index(nx, ny, grid_width)] {
                let q = samples[sample_idx];
                let dx = p.x - q.x;
                let dy = p.y - q.y;
                let d2 = dx * dx + dy * dy;

                if d2 < r2 {
                    return false;
                }
            }
        }
    }

    true
}

pub fn grid_coords(p: Point2, cell_size: f32) -> (usize, usize) {
    let gx = (p.x / cell_size).floor() as usize;
    let gy = (p.y / cell_size).floor() as usize;
    (gx, gy)
}

fn grid_index(x: usize, y: usize, grid_width: usize) -> usize {
    y * grid_width + x
}


#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

pub fn poisson_disc_bridson_3d<R: Rng + ?Sized>(
    width: f32,
    height: f32,
    depth: f32,
    r: f32,
    k: usize,
    rng: &mut R,
) -> Vec<Point3> {
    assert!(width > 0.0);
    assert!(height > 0.0);
    assert!(depth > 0.0);
    assert!(r > 0.0);
    assert!(k > 0);

    let cell_size = r / 3.0_f32.sqrt();

    let nx = (width / cell_size).ceil() as usize;
    let ny = (height / cell_size).ceil() as usize;
    let nz = (depth / cell_size).ceil() as usize;

    let mut grid: Vec<Option<usize>> = vec![None; nx * ny * nz];

    let mut samples = Vec::<Point3>::new();
    let mut active = Vec::<usize>::new();

    let p0 = Point3 {
        x: rng.random_range(0.0..width),
        y: rng.random_range(0.0..height),
        z: rng.random_range(0.0..depth),
    };

    samples.push(p0);
    active.push(0);

    let (gx, gy, gz) = grid_coords_3d(p0, cell_size);
    grid[grid_index_3d(gx, gy, gz, nx, ny)] = Some(0);

    while !active.is_empty() {
        let active_i = rng.random_range(0..active.len());
        let sample_idx = active[active_i];
        let base = samples[sample_idx];

        let mut found = false;

        for _ in 0..k {
            let candidate = generate_shell_candidate(base, r, rng);

            if candidate.x < 0.0
                || candidate.x >= width
                || candidate.y < 0.0
                || candidate.y >= height
                || candidate.z < 0.0
                || candidate.z >= depth
            {
                continue;
            }

            if is_valid_candidate_3d(
                candidate,
                &samples,
                &grid,
                nx,
                ny,
                nz,
                cell_size,
                r,
            ) {
                let new_idx = samples.len();
                samples.push(candidate);
                active.push(new_idx);

                let (cgx, cgy, cgz) = grid_coords_3d(candidate, cell_size);
                grid[grid_index_3d(cgx, cgy, cgz, nx, ny)] = Some(new_idx);

                found = true;
                break;
            }
        }

        if !found {
            active.swap_remove(active_i);
        }
    }

    samples
}

fn generate_shell_candidate<R: Rng + ?Sized>(
    base: Point3,
    r: f32,
    rng: &mut R,
) -> Point3 {
    // Uniform direction on the sphere.
    let u = rng.random_range(-1.0..1.0);
    let theta = rng.random_range(0.0..(2.0 * PI));

    let s = (1.0_f32 - u * u).sqrt();

    let dx = s * theta.cos();
    let dy = s * theta.sin();
    let dz = u;

    // Radius uniformly by volume in spherical shell [r, 2r].
    //
    // radius^3 ~ Uniform(r^3, (2r)^3) = Uniform(r^3, 8r^3)
    let t:f32 = rng.random_range(0.0..1.0);
    let radius = r * (1.0 + 7.0 * t).cbrt();

    Point3 {
        x: base.x + radius * dx,
        y: base.y + radius * dy,
        z: base.z + radius * dz,
    }
}

fn is_valid_candidate_3d(
    p: Point3,
    samples: &[Point3],
    grid: &[Option<usize>],
    nx: usize,
    ny: usize,
    nz: usize,
    cell_size: f32,
    r: f32,
) -> bool {
    let (gx, gy, gz) = grid_coords_3d(p, cell_size);

    let x0 = gx.saturating_sub(2);
    let y0 = gy.saturating_sub(2);
    let z0 = gz.saturating_sub(2);

    let x1 = (gx + 2).min(nx - 1);
    let y1 = (gy + 2).min(ny - 1);
    let z1 = (gz + 2).min(nz - 1);

    let r2 = r * r;

    for zz in z0..=z1 {
        for yy in y0..=y1 {
            for xx in x0..=x1 {
                if let Some(sample_idx) = grid[grid_index_3d(xx, yy, zz, nx, ny)] {
                    let q = samples[sample_idx];

                    let dx = p.x - q.x;
                    let dy = p.y - q.y;
                    let dz = p.z - q.z;

                    if dx * dx + dy * dy + dz * dz < r2 {
                        return false;
                    }
                }
            }
        }
    }

    true
}

fn grid_coords_3d(p: Point3, cell_size: f32) -> (usize, usize, usize) {
    (
        (p.x / cell_size).floor() as usize,
        (p.y / cell_size).floor() as usize,
        (p.z / cell_size).floor() as usize,
    )
}

fn grid_index_3d(x: usize, y: usize, z: usize, nx: usize, ny: usize) -> usize {
    x + nx * (y + ny * z)
}