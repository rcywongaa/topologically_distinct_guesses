use ndarray::{Array, Array1, Dim};
use num_traits::Float;

pub fn linspace<A: Float>(start: Array1<A>, end: Array1<A>, n: usize) -> Array<A, Dim<[usize; 2]>> {
  assert!(start.len() == end.len());
  Array::from_shape_vec(
    (start.len(), n),
    (0..start.len())
      .flat_map(|idx| Array::linspace(start[idx], end[idx], n))
      .collect(),
  )
  .unwrap()
  .reversed_axes()
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  #[test]
  fn test_linspace() {
    assert_eq!(
      linspace(array![1., 2., 3.], array![4., 5., 6.], 4),
      array![[1., 2., 3.], [2., 3., 4.], [3., 4., 5.], [4., 5., 6.]]
    )
  }
}
