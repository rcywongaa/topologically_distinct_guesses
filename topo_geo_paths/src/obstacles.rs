#![allow(dead_code)]
use std::rc::Rc;

use ndarray::Array;
use ndarray_linalg::Norm;

pub fn combine_obstacles<const T: usize>(
  obs_func: Vec<Rc<dyn Fn([f32; T]) -> bool>>,
) -> impl Fn([f32; T]) -> bool {
  move |x: [f32; T]| obs_func.iter().any(|func| func(x))
}

pub fn sphere_obstacle<const T: usize>(center: [f32; T], radius: f32) -> impl Fn([f32; T]) -> bool {
  move |pos: [f32; T]| (Array::from_iter(center) - Array::from_iter(pos)).norm_l2() <= radius
}

pub fn aabb_obstacle<const T: usize>(center: [f32; T], dim: [f32; T]) -> impl Fn([f32; T]) -> bool {
  move |pos: [f32; T]| {
    for i in 0..T {
      if (pos[i] - center[i]).abs() > dim[i] {
        return false;
      }
    }
    true
  }
}

pub fn cylinder_obstacle(center: [f32; 3], radius_height: [f32; 2]) -> impl Fn([f32; 3]) -> bool {
  move |pos: [f32; 3]| {
    let radius = radius_height[0];
    let height = radius_height[1];
    if pos[2] > center[2] + height || pos[2] < center[2] - height {
      return false;
    }
    sphere_obstacle([center[0], center[1]], radius)([pos[0], pos[1]])
  }
}

pub fn to_rc<const T: usize>(
  obs: impl Fn([f32; T]) -> bool + 'static,
) -> Rc<dyn Fn([f32; T]) -> bool> {
  Rc::new(obs)
}
