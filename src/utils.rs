pub fn trapz(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len(), "x and y must have same length");
    assert!(x.len() >= 2, "need at least 2 points");

    let mut sum = 0.0;

    for i in 0..(x.len() - 1) {
        let dx = x[i + 1] - x[i];
        sum += 0.5 * dx * (y[i] + y[i + 1]);
    }

    sum
}

pub fn cumtrapz(x: &[f32], y: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), y.len(), "x and y must have same length");
    assert_eq!(x.len(), out.len(), "output must match input length");
    assert!(x.len() >= 2, "need at least 2 points");

    out[0] = 0.0;

    let mut acc = 0.0;

    for i in 0..(x.len() - 1) {
        let dx = x[i + 1] - x[i];
        acc += 0.5 * dx * (y[i] + y[i + 1]);
        out[i + 1] = acc;
    }
}

pub fn interp1(x: &[f32], y: &[f32], xi: &[f32], yi: &mut [f32]) {
    assert_eq!(x.len(), y.len(), "x and y must have same length");
    assert_eq!(xi.len(), yi.len(), "xi and yi must have same length");
    assert!(x.len() >= 2, "need at least 2 data points");

    let n = x.len();

    for (j, &xq) in xi.iter().enumerate() {
        // Handle boundaries (clamp like MATLAB default behavior)
        if xq <= x[0] {
            yi[j] = y[0];
            continue;
        }
        if xq >= x[n - 1] {
            yi[j] = y[n - 1];
            continue;
        }

        // Binary search to find interval
        let mut lo = 0;
        let mut hi = n - 1;

        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if x[mid] <= xq {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        // Now x[lo] <= xq <= x[lo+1]
        let x0 = x[lo];
        let x1 = x[lo + 1];
        let y0 = y[lo];
        let y1 = y[lo + 1];

        let t = (xq - x0) / (x1 - x0);
        yi[j] = y0 + t * (y1 - y0);
    }
}

