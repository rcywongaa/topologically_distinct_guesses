use std::{fs, path::Path};

use constcat::concat;
use once_cell::sync::Lazy;
use pyo3::{
  prelude::*,
  types::{IntoPyDict, PyList, PyModule},
};

pub struct PlanningSetting {
  pub sphere_obstacle_specs: Vec<([f32; 3], f32)>,
  pub aabb_obstacle_specs: Vec<([f32; 3], [f32; 3])>,
  pub cylinder_obstacle_specs: Vec<([f32; 3], [f32; 2])>,
  pub eef_traj_func: fn(f32) -> [f32; 3],
  pub forearm_length: f32,
  pub upperarm_length: f32,
  pub arm_base_height: f32,
  pub mobile_base_height: f32,
  pub initial_x_b: [f32; 3],
  pub initial_x_w: [f32; 3],
  pub initial_x_e: [f32; 3],
  pub final_x_b: [f32; 3],
  pub final_x_w: [f32; 3],
  pub final_x_e: [f32; 3],
  pub num_paths: usize,
  pub link_radius: f32,
  pub base_radius: f32,
}

pub fn get_planning_setting() -> PlanningSetting {
  let (initial_x_e, initial_x_w, initial_x_b) = get_pose_x("initial");
  let (final_x_e, final_x_w, final_x_b) = get_pose_x("final");
  PlanningSetting {
    sphere_obstacle_specs: get_sphere_obstacle_specs(),
    aabb_obstacle_specs: get_aabb_obstacle_specs(),
    cylinder_obstacle_specs: get_cylinder_obstacle_specs(),
    eef_traj_func: get_eef_traj_func,
    forearm_length: get_f32_constant("forearm_length"),
    upperarm_length: get_f32_constant("upperarm_length"),
    arm_base_height: get_f32_constant("arm_base_height"),
    mobile_base_height: get_f32_constant("mobile_base_height"),
    initial_x_b,
    initial_x_w,
    initial_x_e,
    final_x_b,
    final_x_w,
    final_x_e,
    num_paths: get_usize_constant("num_paths"),
    link_radius: get_f32_constant("link_radius"),
    base_radius: get_f32_constant("base_radius"),
  }
}

const PLANNING_SETTING_PY_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../config");
const PLANNING_SETTING_PY_CONTENT: &str = include_str!(std::concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/../config/planning_setting.py"
));

/* From https://pyo3.rs/v0.21.1/python-from-rust/calling-existing-code#include-multiple-python-files */
fn import_dependencies(py: &Python) {
  let path = Path::new(PLANNING_SETTING_PY_DIR);
  let syspath = py
    .import_bound("sys")
    .unwrap()
    .getattr("path")
    .unwrap()
    .downcast_into::<PyList>()
    .unwrap();
  syspath.insert(0, &path).unwrap();
}

fn get_attr(py: Python, attr_name: &str) -> Py<PyAny> {
  PyModule::from_code_bound(
    py,
    PLANNING_SETTING_PY_CONTENT,
    "config.planning_setting",
    "config.planning_setting",
  )
  .unwrap()
  .getattr(attr_name)
  .unwrap()
  .into()
}

fn get_eef_traj_func(t: f32) -> [f32; 3] {
  Python::with_gil(|py| -> [f32; 3] {
    import_dependencies(&py);
    let is_in_obstacle_py: Py<PyAny> = get_attr(py, "get_eef_position");
    is_in_obstacle_py
      .call_bound(py, (), Some(&[("t", t)].into_py_dict_bound(py)))
      .as_ref()
      .unwrap()
      .extract(py)
      .unwrap()
  })
}

fn get_f32_constant(name: &str) -> f32 {
  Python::with_gil(|py| -> f32 {
    import_dependencies(&py);
    let constant: Py<PyAny> = get_attr(py, name);
    constant.extract(py).unwrap()
  })
}

fn get_usize_constant(name: &str) -> usize {
  Python::with_gil(|py| -> usize {
    import_dependencies(&py);
    let constant: Py<PyAny> = get_attr(py, name);
    constant.extract(py).unwrap()
  })
}

fn get_vec3_constant(name: &str) -> [f32; 3] {
  Python::with_gil(|py| -> [f32; 3] {
    import_dependencies(&py);
    let constant: Py<PyAny> = get_attr(py, name);
    constant.extract(py).unwrap()
  })
}

fn get_pose_x(prefix: &str) -> ([f32; 3], [f32; 3], [f32; 3]) {
  let x_e = get_vec3_constant(&format!("{prefix}_x_e"));
  let x_w = get_vec3_constant(&format!("{prefix}_x_w"));
  let x_b = get_vec3_constant(&format!("{prefix}_x_b"));
  (x_e, x_w, x_b)
}

fn get_sphere_obstacle_specs() -> Vec<([f32; 3], f32)> {
  Python::with_gil(|py| -> Vec<([f32; 3], f32)> {
    import_dependencies(&py);
    let sphere_obstacle_specs: Py<PyAny> = get_attr(py, "sphere_obstacle_specs");
    sphere_obstacle_specs.extract(py).unwrap()
  })
}

fn get_aabb_obstacle_specs() -> Vec<([f32; 3], [f32; 3])> {
  Python::with_gil(|py| -> Vec<([f32; 3], [f32; 3])> {
    import_dependencies(&py);
    let obstacle_specs: Py<PyAny> = get_attr(py, "aabb_obstacle_specs");
    obstacle_specs.extract(py).unwrap()
  })
}

fn get_cylinder_obstacle_specs() -> Vec<([f32; 3], [f32; 2])> {
  Python::with_gil(|py| -> Vec<([f32; 3], [f32; 2])> {
    import_dependencies(&py);
    let obstacle_specs: Py<PyAny> = get_attr(py, "cylinder_obstacle_specs");
    obstacle_specs.extract(py).unwrap()
  })
}

/**
 * The performance of this is probably too low for use in generating the graph...
 */
#[deprecated(note = "This has poor performance")]
fn is_in_obstacle(sample: [f32; 3]) -> bool {
  Python::with_gil(|py| -> bool {
    import_dependencies(&py);
    let is_in_obstacle_py: Py<PyAny> = get_attr(py, "is_in_obstacle");
    is_in_obstacle_py
      .call_bound(py, (), Some(&[("sample", sample)].into_py_dict_bound(py)))
      .as_ref()
      .unwrap()
      .extract(py)
      .unwrap()
  })
}

#[cfg(test)]
mod tests {
  use std::time::Instant;

  // Note this useful idiom: importing names from outer (for mod tests) scope.
  use super::*;

  #[allow(deprecated)]
  #[test]
  fn test_is_in_obstacle() {
    let sample = [0.5, 0.0, 0.0];
    println!("{:?} is in obstacle? {}", sample, is_in_obstacle(sample));
  }

  #[allow(deprecated)]
  #[test]
  fn test_is_in_obstacle_performance() {
    let now = Instant::now();
    for _ in 1..1000 {
      is_in_obstacle([0.5, 0.0, 0.0]);
    }
    println!("Time took for 1000 calls: {}ms", now.elapsed().as_millis());
  }

  #[test]
  fn test_get_sphere_obstacle_specs() {
    println!("ShereObstacle specs: {:?}", get_sphere_obstacle_specs());
  }
}
