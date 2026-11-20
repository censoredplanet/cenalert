//! Royston's AS R94 `swilk` routine — the Shapiro-Wilk W statistic and its
//! p-value, transcribed from the SciPy Cython port (`_ansari_swilk_statistics`).
//!
//! Constants come verbatim from Royston P., "Remark AS R94: A Remark on
//! Algorithm AS 181: The W-test for Normality", 1995, Applied Statistics 44,
//! and match SciPy 1.17.1's compiled routine bit-for-bit. The p-value uses AS 66
//! (`alnorm`) and the normal quantile uses AS 111 (`ppnd`) exactly as SciPy
//! does — not the Cephes `ndtr`/`ndtri`, which would diverge in the last bits.
//!
//! SciPy's `swilk` is compiled from Cython by a C compiler that contracts
//! `a*b + c` into a fused multiply-add (clang's default `-ffp-contract=fast`).
//! Rust does not contract by default, so the polynomial and accumulation steps
//! that SciPy fuses are written with explicit `f64::mul_add` here. Without this
//! the weights and W drift by 1-2 ULP from SciPy; with it, W and the p-value are
//! bit-identical.

// The polynomial constants are transcribed at the source's literal precision.
#![allow(clippy::excessive_precision)]

const C1: [f64; 6] = [
    0.,
    0.221157,
    -0.147981,
    -0.207119e1,
    0.4434685e1,
    -0.2706056e1,
];
const C2: [f64; 6] = [
    0.,
    0.42981e-1,
    -0.293762,
    -0.1752461e1,
    0.5682633e1,
    -0.3582633e1,
];
const C3: [f64; 4] = [0.5440, -0.39978, 0.25054e-1, -0.6714e-3];
const C4: [f64; 4] = [0.13822e1, -0.77857, 0.62767e-1, -0.20322e-2];
const C5: [f64; 4] = [-0.15861e1, -0.31082, -0.83751e-1, 0.38915e-2];
const C6: [f64; 3] = [-0.4803, -0.82676e-1, 0.30302e-2];
const G: [f64; 2] = [-0.2273e1, 0.459];

const SQRTH: f64 = std::f64::consts::FRAC_1_SQRT_2;
const SMALL: f64 = 1e-19;

/// Outcome of [`swilk`]: W, its p-value, and Royston's `ifault` code.
///
/// `ifault` mirrors SciPy: 0 normal, 2 for n > 5000 (W accurate, p-value may
/// not be), 6 for zero range. SciPy's `shapiro` accepts 0 and 2 silently and
/// warns otherwise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Swilk {
    pub w: f64,
    pub pw: f64,
    pub ifault: i32,
}

/// Shapiro-Wilk W and its significance for an ascending-sorted sample `x`.
///
/// `x` must already be sorted ascending (SciPy sorts and then subtracts a
/// near-median element; the W statistic is location-invariant so any constant
/// shift is irrelevant here). `n2 = n/2` weight slots are computed internally.
#[must_use]
pub fn swilk(x: &[f64]) -> Swilk {
    let n = x.len();
    let nn2 = n / 2;
    let n2 = nn2;

    if n < 3 {
        return Swilk {
            w: 1.,
            pw: 1.,
            ifault: 1,
        };
    }

    let an = n as f64;
    let mut a = vec![0.0_f64; n2];

    if n == 3 {
        a[0] = SQRTH;
    } else {
        let an25 = an + 0.25;
        let mut summ2 = 0.0;
        #[allow(clippy::needless_range_loop)]
        for ind1 in 0..n2 {
            let temp = ppnd((ind1 as f64 + 1.0 - 0.375) / an25);
            a[ind1] = temp;
            summ2 = temp.mul_add(temp, summ2);
        }
        summ2 *= 2.0;
        let ssumm2 = summ2.sqrt();
        let rsn = 1.0 / an.sqrt();
        let a1 = poly(&C1, rsn) - (a[0] / ssumm2);

        let (i1, fac);
        if n > 5 {
            i1 = 2;
            let a2 = -a[1] / ssumm2 + poly(&C2, rsn);
            fac = ((summ2 - (2.0 * a[0] * a[0]) - 2.0 * a[1] * a[1])
                / (1.0 - (2.0 * a1 * a1) - 2.0 * a2 * a2))
                .sqrt();
            a[1] = a2;
        } else {
            i1 = 1;
            fac = ((summ2 - 2.0 * a[0] * a[0]) / (1.0 - 2.0 * a1 * a1)).sqrt();
        }
        a[0] = a1;
        for ai in a.iter_mut().take(nn2).skip(i1) {
            *ai *= -1.0 / fac;
        }
    }

    let n1 = n;
    let range = x[n1 - 1] - x[0];
    if range < SMALL {
        return Swilk {
            w: 1.,
            pw: 1.,
            ifault: 6,
        };
    }

    // Centre x and the weights, then form the correlation between them; W is
    // one minus the squared residual of the regression of x on a.
    let mut sx = x[0] / range;
    let mut sa = -a[0];
    let mut ind2 = n - 2;
    // `ind1` walks up while `ind2` walks down so the antisymmetric weight
    // vector is reflected at the midpoint; the two indices must move together.
    #[allow(clippy::needless_range_loop)]
    for ind1 in 1..n1 {
        let xi = x[ind1] / range;
        sx += xi;
        if ind1 != ind2 {
            sa += signed_a(&a, ind1, ind2);
        }
        ind2 = ind2.wrapping_sub(1);
    }

    let ifault = if n > 5000 { 2 } else { 0 };

    sa /= n1 as f64;
    sx /= n1 as f64;
    let (mut ssa, mut ssx, mut sax) = (0.0, 0.0, 0.0);
    let mut ind2 = n - 1;
    #[allow(clippy::needless_range_loop)]
    for ind1 in 0..n1 {
        let asa = if ind1 != ind2 {
            signed_a(&a, ind1, ind2) - sa
        } else {
            -sa
        };
        let xsx = x[ind1] / range - sx;
        ssa = asa.mul_add(asa, ssa);
        ssx = xsx.mul_add(xsx, ssx);
        sax = asa.mul_add(xsx, sax);
        ind2 = ind2.wrapping_sub(1);
    }

    let ssassx = (ssa * ssx).sqrt();
    let w1 = (ssassx - sax) * (ssassx + sax) / (ssa * ssx);
    let w = 1.0 - w1;

    if n == 3 {
        if w < 0.75 {
            return Swilk {
                w: 0.75,
                pw: 0.,
                ifault,
            };
        }
        let pi6 = 6.0 / std::f64::consts::PI;
        let pw = (-pi6).mul_add(w.sqrt().acos(), 1.0);
        return Swilk { w, pw, ifault };
    }

    let mut y = w1.ln();
    let xx = an.ln();
    let (m, s);
    if n <= 11 {
        let gamma = poly(&G, an);
        if y >= gamma {
            return Swilk {
                w,
                pw: SMALL,
                ifault,
            };
        }
        y = -(gamma - y).ln();
        m = poly(&C3, an);
        s = poly(&C4, an).exp();
    } else {
        m = poly(&C5, xx);
        s = poly(&C6, xx).exp();
    }

    let pw = alnorm((y - m) / s, true);
    Swilk { w, pw, ifault }
}

/// The signed weight picked up at position `ind1` while sweeping symmetric pairs.
///
/// AS R94 stores only the first half of the antisymmetric weight vector; the
/// upper half is the negated mirror. `ind2` walks down from the top so the lower
/// index always selects the stored slot and the sign flips at the midpoint.
#[inline]
fn signed_a(a: &[f64], ind1: usize, ind2: usize) -> f64 {
    let sign = if ind1 < ind2 { -1.0 } else { 1.0 };
    sign * a[ind1.min(ind2)]
}

/// Normal quantile, AS 111 (`ppnd`, Beasley-Springer 1977). Transcribed exactly
/// from SciPy's `_ppnd`; this is what makes the weights bit-identical to SciPy.
fn ppnd(p: f64) -> f64 {
    const A0: f64 = 2.50662823884;
    const A1: f64 = -18.61500062529;
    const A2: f64 = 41.39119773534;
    const A3: f64 = -25.44106049637;
    const B1: f64 = -8.47351093090;
    const B2: f64 = 23.08336743743;
    const B3: f64 = -21.06224101826;
    const B4: f64 = 3.13082909833;
    const C0: f64 = -2.78718931138;
    const C1P: f64 = -2.29796479134;
    const C2P: f64 = 4.85014127135;
    const C3P: f64 = 2.32121276858;
    const D1: f64 = 3.54388924762;
    const D2: f64 = 1.63706781897;
    const SPLIT: f64 = 0.42;

    let q = p - 0.5;
    if q.abs() <= SPLIT {
        let r = q * q;
        let temp = q * A3.mul_add(r, A2).mul_add(r, A1).mul_add(r, A0);
        return temp
            / B4.mul_add(r, B3)
                .mul_add(r, B2)
                .mul_add(r, B1)
                .mul_add(r, 1.0);
    }

    let mut r = p;
    if q > 0.0 {
        r = 1.0 - p;
    }
    if r > 0.0 {
        r = (-r.ln()).sqrt();
    } else {
        return 0.0;
    }
    let temp =
        C3P.mul_add(r, C2P).mul_add(r, C1P).mul_add(r, C0) / D2.mul_add(r, D1).mul_add(r, 1.0);
    if q < 0.0 { -temp } else { temp }
}

/// Upper-tail normal probability, AS 66 (`alnorm`, Hill 1973). Transcribed
/// exactly from SciPy's `_alnorm`; SciPy's swilk uses this for the p-value, not
/// Cephes `ndtr`, so we must too.
fn alnorm(x: f64, upper: bool) -> f64 {
    const A1: f64 = 0.398942280444;
    const A2: f64 = 0.399903438504;
    const A3: f64 = 5.75885480458;
    const A4: f64 = 29.8213557808;
    const A5: f64 = 2.62433121679;
    const A6: f64 = 48.6959930692;
    const A7: f64 = 5.92885724438;
    const B1: f64 = 0.398942280385;
    const B2: f64 = 3.8052e-8;
    const B3: f64 = 1.00000615302;
    const B4: f64 = 3.98064794e-4;
    const B5: f64 = 1.98615381364;
    const B6: f64 = 0.151679116635;
    const B7: f64 = 5.29330324926;
    const B8: f64 = 4.8385912808;
    const B9: f64 = 15.1508972451;
    const B10: f64 = 0.742380924027;
    const B11: f64 = 30.789933034;
    const B12: f64 = 3.99019417011;
    const LTONE: f64 = 7.0;
    const UTZERO: f64 = 38.0;
    const CON: f64 = 1.28;

    let mut upper = upper;
    let mut z = x;
    // AS 66 folds at zero with `not (z > 0)`, so a NaN (or signed zero) takes
    // the negate branch exactly as SciPy's routine does.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(z > 0.0) {
        upper = !upper;
        z = -z;
    }
    if !((z <= LTONE) || (upper && z <= UTZERO)) {
        return if upper { 0.0 } else { 1.0 };
    }
    let y = 0.5 * z * z;
    let temp = if z <= CON {
        z.mul_add(
            -(A1 - A2 * y / (y + A3 - A4 / (y + A5 + A6 / (y + A7)))),
            0.5,
        )
    } else {
        B1 * (-y).exp()
            / (z - B2
                + B3 / (z + B4 + B5 / (z - B6 + B7 / (z + B8 - B9 / (z + B10 + B11 / (z + B12))))))
    };
    if upper { temp } else { 1.0 - temp }
}

/// Horner polynomial `c[0] + x·(c[1] + x·(…))`, matching AS R94's `poly`.
#[inline]
fn poly(c: &[f64], x: f64) -> f64 {
    let nord = c.len();
    let mut res = c[0];
    if nord == 1 {
        return res;
    }
    let mut p = x * c[nord - 1];
    if nord == 2 {
        return res + p;
    }
    for ind in (1..nord - 1).rev() {
        p = (p + c[ind]) * x;
    }
    res += p;
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppnd_median_is_zero() {
        assert_eq!(ppnd(0.5), 0.0);
    }

    #[test]
    fn ppnd_symmetric() {
        // AS 111 is antisymmetric about p=0.5 by construction.
        for p in [0.1, 0.25, 0.4, 0.7, 0.9] {
            let a = ppnd(p);
            let b = ppnd(1.0 - p);
            assert!(
                (a + b).abs() < 1e-12,
                "ppnd({p}) = {a}, ppnd({}) = {b}",
                1.0 - p
            );
        }
    }

    #[test]
    fn poly_matches_horner() {
        let c = [1.0, 2.0, 3.0, 4.0];
        // 1 + 2x + 3x^2 + 4x^3 at x=2 = 1+4+12+32 = 49
        assert_eq!(poly(&c, 2.0), 49.0);
    }

    #[test]
    fn n3_floor() {
        // Sorted, median-shifted [5,5,9] → [0,0,4]; W drops below 0.75 and is
        // clamped to the floor with p=0, matching scipy.
        let r = swilk(&[0.0, 0.0, 4.0]);
        assert_eq!(r.w, 0.75);
        assert_eq!(r.pw, 0.0);
    }
}
