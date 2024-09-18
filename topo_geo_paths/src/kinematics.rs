use nalgebra::{distance, Point3, Rotation3, Vector3};
use nshare::{ToNalgebra, ToNdarray1};

use crate::not_nan_util::{dist, from_not_nan_1d, norm, to_not_nan_1d};

pub fn calc_elbow_position(
  forearm_length: f32,
  upperarm_length: f32,
  eef: &[f32; 3],
  base: &[f32; 3],
  is_up: bool,
) -> Option<[f32; 3]> {
  let l1 = upperarm_length;
  let A = Point3::new(base[0], base[1], base[2]);
  let l2 = forearm_length;
  let B = Point3::new(eef[0], eef[1], eef[2]);
  let l3 = distance(&A, &B);
  let B_proj = Point3::new(B[0], B[1], A[2]);
  let AB_proj = B_proj - A;

  /* https://math.stackexchange.com/a/3021334 */
  let phi1 = (B[2] - A[2]).atan2(AB_proj.norm());
  if phi1.is_nan() {
    return None;
  }
  let mut phi2 = ((l1.powf(2.0) + l3.powf(2.) - l2.powf(2.)) / (2.0 * l1 * l3)).acos();
  if phi2.is_nan() {
    return None;
  }
  if !is_up {
    phi2 = -phi2
  }
  let C = A + (AB_proj.normalize() * (phi1 + phi2).cos() + Vector3::z() * (phi1 + phi2).sin()) * l1;
  Some([C[0], C[1], C[2]])
}

#[cfg(test)]
mod tests {

  use approx::assert_abs_diff_eq;

  use super::*;

  #[test]
  fn test_calc_elbow_position() {
    assert_abs_diff_eq!(
      calc_elbow_position(1.0, 0.5, &[0.2, 1.0, 0.5], &[0.2, 0.0, 0.0], true)
        .unwrap()
        .as_ref(),
      [0.2, 0.0, 0.5].as_ref(),
      epsilon = 1e-6
    )
  }
}
