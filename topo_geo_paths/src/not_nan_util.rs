use ndarray::{Array, Array1, Ix1};
use ndarray_linalg::Norm;
use num_traits::ToPrimitive;
use ordered_float::{FloatCore, NotNan};

pub type NotNan1D = Array<NotNan<f32>, Ix1>;

pub fn norm(p: &NotNan1D) -> f32 {
  let p_vec = p.map(|x| x.into_inner());
  p_vec.norm_l2()
}

pub fn dist(p1: &NotNan1D, p2: &NotNan1D) -> f32 {
  norm(&(p1 - p2))
}

pub fn to_not_nan<T: FloatCore>(x: T) -> NotNan<T> {
  NotNan::<T>::new(x).unwrap()
}

pub fn to_not_nan_1d(arr: &[f32]) -> NotNan1D {
  Array1::from_iter(arr.iter().map(|x| to_not_nan(*x)))
}

pub fn from_not_nan_1d<const N: usize>(notnan1d: &NotNan1D) -> [f32; N] {
  notnan1d
    .iter()
    .map(|&notnan| notnan.to_f32().unwrap())
    .collect::<Vec<f32>>()
    .try_into()
    .unwrap()
}
