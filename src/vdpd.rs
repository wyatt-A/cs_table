use rand::prelude::{SliceRandom, StdRng};
use rand::SeedableRng;

#[derive(Clone, Copy, Debug)]
struct Pt {
    x: f32,
    y: f32,
    iy: usize,
    iz: usize,
}

fn radius(x: f32, y: f32, gamma: f32, delta: f32) -> f32 {
    let r = (x * x + y * y).sqrt();
    (r + delta) / gamma
}

pub fn generate_vdpd_mask(
    ny: usize,
    nz: usize,
    accel: f32,
    fully_sampled_radius: f32,
    gamma: f32,
    delta: f32,
    seed: u64,
) -> Vec<bool> {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut candidates = Vec::with_capacity(ny * nz);

    for iz in 0..nz {
        for iy in 0..ny {
            let y = (iy as f32 + 0.5) / ny as f32 - 0.5;
            let z = (iz as f32 + 0.5) / nz as f32 - 0.5;

            candidates.push(Pt {
                x: y,
                y: z,
                iy,
                iz,
            });
        }
    }

    candidates.shuffle(&mut rng);

    let min_r = radius(0.0, 0.0, gamma, delta);
    let cell_size = min_r / 2.0_f32.sqrt();

    let grid_w = (1.0 / cell_size).ceil() as usize + 1;
    let grid_h = grid_w;

    let mut spatial_grid: Vec<Vec<Pt>> = vec![Vec::new(); grid_w * grid_h];
    let mut mask = vec![false; ny * nz];

    let target_samples = ((ny * nz) as f32 / accel).round() as usize;
    let mut n_samples = 0usize;

    for p in candidates {
        if n_samples >= target_samples {
            break;
        }

        let rr = (p.x * p.x + p.y * p.y).sqrt();

        // Always keep a calibration region.
        if rr <= fully_sampled_radius {
            let idx = p.iy + p.iz * ny;
            if !mask[idx] {
                mask[idx] = true;
                n_samples += 1;
            }
            continue;
        }

        let rp = radius(p.x, p.y, gamma, delta);

        let gx = ((p.x + 0.5) / cell_size).floor() as isize;
        let gy = ((p.y + 0.5) / cell_size).floor() as isize;

        let search = (rp / cell_size).ceil() as isize + 2;
        let mut ok = true;

        'outer: for dy in -search..=search {
            for dx in -search..=search {
                let xx = gx + dx;
                let yy = gy + dy;

                if xx < 0 || yy < 0 || xx >= grid_w as isize || yy >= grid_h as isize {
                    continue;
                }

                let cell = &spatial_grid[xx as usize + yy as usize * grid_w];

                for q in cell {
                    let rq = radius(q.x, q.y, gamma, delta);
                    let min_dist = rp.max(rq);

                    let ddx = p.x - q.x;
                    let ddy = p.y - q.y;
                    let d2 = ddx * ddx + ddy * ddy;

                    if d2 < min_dist * min_dist {
                        ok = false;
                        break 'outer;
                    }
                }
            }
        }

        if ok {
            let idx = p.iy + p.iz * ny;
            mask[idx] = true;

            let cell_idx = gx as usize + gy as usize * grid_w;
            spatial_grid[cell_idx].push(p);

            n_samples += 1;
        }
    }

    mask
}