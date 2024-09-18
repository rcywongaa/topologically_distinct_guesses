#![allow(dead_code)]
use std::{
  cmp::{max, min},
  iter::zip,
  rc::Rc,
};

use nalgebra::Vector3;
use num_traits::FromPrimitive;
use ordered_float::NotNan;

use crate::not_nan_util::to_not_nan_1d;

pub fn combine_link_obstacles<const T: usize>(
  /*
    We cannot use Box because we may not want to own the obs_funcs
    We cannot Clone/Copy obs_funcs because Clone/Copy is not object safe
    We also want the returned function to potentially outlive the obs_funcs
    So instead, we make it a reference counted (shared ownership) object
    See https://stackoverflow.com/q/30353462/3177701
  */
  obs_funcs: Vec<Rc<dyn Fn([f32; T], [f32; T]) -> bool>>,
) -> impl Fn([f32; T], [f32; T]) -> bool {
  move |x: [f32; T], y: [f32; T]| obs_funcs.iter().any(|func| func(x, y))
}

pub fn to_rc<const T: usize>(
  obs: impl Fn([f32; T], [f32; T]) -> bool + 'static,
) -> Rc<dyn Fn([f32; T], [f32; T]) -> bool> {
  Rc::new(obs)
}

/**
 * Returns an obstacle function
 * where
 * the first argument represents the sample position
 * the second argument represents the center
 */
pub fn link_sphere_obstacle(center: [f32; 3], radius: f32) -> impl Fn([f32; 3], [f32; 3]) -> bool {
  // from https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
  move |start: [f32; 3], end: [f32; 3]| {
    let x1 = Vector3::from_iterator(start);
    let x2 = Vector3::from_iterator(end);
    let x0 = Vector3::from_iterator(center);
    let t = -(x1 - x0).dot(&(x2 - x1)) / (x2 - x1).norm_squared();
    if t > 1. {
      (x2 - x0).norm_squared() <= radius.powi(2)
    } else if t < 0. {
      return (x1 - x0).norm_squared() <= radius.powi(2);
    } else {
      return ((x2 - x1).cross(&(x1 - x0))).norm_squared() / (x2 - x1).norm_squared()
        <= radius.powi(2);
    }
  }
}

pub fn link_aabb_obstacle(center: [f32; 3], dim: [f32; 3]) -> impl Fn([f32; 3], [f32; 3]) -> bool {
  // From https://tavianator.com/2022/ray_box_boundary.html
  let box_min = to_not_nan_1d(&[
    center[0] - dim[0] / 2.0,
    center[1] - dim[1] / 2.0,
    center[2] - dim[2] / 2.0,
  ]);
  let box_max = to_not_nan_1d(&[
    center[0] + dim[0] / 2.0,
    center[1] + dim[1] / 2.0,
    center[2] + dim[2] / 2.0,
  ]);
  move |start: [f32; 3], end: [f32; 3]| {
    let start = to_not_nan_1d(&start);
    let end = to_not_nan_1d(&end);
    let ray_origin = &start;
    let ray_dir = &end - &start;
    let t1 = (&box_min - ray_origin) / &ray_dir;
    let t2 = (&box_max - ray_origin) / &ray_dir;

    let mut tmin = NotNan::from_f32(0.0).unwrap();
    let mut tmax = NotNan::from_f32(1.0).unwrap();
    for (t1_ele, t2_ele) in zip(t1.iter(), t2.iter()) {
      tmin = max(tmin, *min(t1_ele, t2_ele));
      tmax = min(tmax, *max(t1_ele, t2_ele));
    }
    tmin < tmax
  }
}

pub fn link_cylinder_obstacle(
  center: [f32; 3],
  radius_height: [f32; 2],
) -> impl Fn([f32; 3], [f32; 3]) -> bool {
  move |start: [f32; 3], end: [f32; 3]| {
    let radius = radius_height[0];
    let height = radius_height[1];
    if start[2] < center[2] - height && end[2] < center[2] - height {
      return false;
    }
    if start[2] > center[2] + height && start[2] > center[2] + height {
      return false;
    }
    link_sphere_obstacle([center[0], center[1], 0.0], radius)(
      [start[0], start[1], 0.0],
      [end[0], end[1], 0.0],
    )
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_link_sphere_obstacle() {
    let obs_func = link_sphere_obstacle([0., 0., 0.], 1.0);
    assert!(obs_func([-1., -1., -1.], [1., 1., 1.]));
    assert!(!obs_func([-1., -1., -1.], [-1., -1., 1.]));
    assert!(obs_func([-1., -1., -0.99], [1., 1., -0.99]));
    assert!(!obs_func([-1., -1., -1.01], [1., 1., -1.01]));
    assert!(!obs_func([2., 2., 2.], [2., 2., 4.]));
    assert!(obs_func([0.99, 0., 0.], [10., 10., 10.]));
    assert!(obs_func([10., 10., 10.], [0., 0.99, 0.]))
  }

  #[test]
  fn test_link_aabb_obstacle() {
    let obs_func = link_aabb_obstacle([0., 1., 0.], [2., 2., 2.]);
    assert!(obs_func([0., 0.1, -1.5], [0., 0.1, 1.5])); // intersect, start outside, end outside
    assert!(obs_func([0., 0.1, -1.], [0., 0.1, 0.])); // intersect, start outside, end inside
    assert!(obs_func([0., 0.1, -0.5], [0., 0.1, 0.5])); // intersect, start inside, end inside
    assert!(obs_func([1.1, 1.9, 0.9], [0.85, 2.1, 1.1])); // intersects near [1., 2., 1.]
    assert!(!obs_func([1.1, 1.9, 0.9], [0.95, 2.1, 1.1])); // no intersect near [1., 2., 1.]

    let obs_func = link_aabb_obstacle([0.0, 0.0, 0.90], [1.5, 0.3, 0.05]);
    assert!(obs_func(
      [0.60856605, -0.114288, 1.01571],
      [0.76923, -0.05128, 0.6438]
    ));
  }
}
