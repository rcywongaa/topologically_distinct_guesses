#![allow(dead_code)]

use ahash::HashSet;
use itertools::Itertools;
use ndarray::{array, concatenate, s, Array, ArrayView, Axis};
use ndarray_rand::rand::{thread_rng, Rng};
use num_traits::ToPrimitive;
use ordered_float::NotNan;
use petgraph::{
  graph::IndexType,
  visit::{IntoNodeIdentifiers, IntoNodeReferences, NodeCount, NodeIndexable},
};
use rand::distributions::Uniform;
use std::{
  cell::Cell,
  cmp::{max, min},
  f32::consts::PI,
  rc::Rc,
  time::Instant,
};
use three_d::{
  degrees, pick, radians, vec3, AmbientLight, Axes, Camera, ClearState, Context, CpuMaterial,
  CpuMesh, DirectionalLight, Event, Geometry, Gm, InnerSpace, InstancedMesh, Instances, Key,
  LogicalPoint, Mat4, Material, Mesh, MetricSpace, Object, OrbitControl, PhysicalMaterial, Quat,
  Srgba, Vec3, Vector3, Viewport, WindowedContext,
};
use winit::{self, platform::run_return::EventLoopExtRunReturn};

use crate::{
  colors,
  graph_utils::{get_physical_path, print_path, unpack_position},
  nag_dijkstra::{get_path_neighborhood, Graph, GraphNode, Nag, NagIndex, NagNode, PhysicalIndex},
  not_nan_util::{to_not_nan_1d, NotNan1D}, // NotNan needed to sort distances in picked_id
};

const DEFAULT_EDGE_ALPHA: u8 = 10;
const DEFAULT_EDGE_WIDTH: f32 = 0.0015;
const COLOR_TRANSFORM_POWER: i32 = 5;
const SPHERE_SIZE: f32 = 0.02;

fn from_not_nan_1d(a: &NotNan1D) -> Vec3 {
  if a.len() == 2 {
    Vec3::new(a[0].into_inner(), a[1].into_inner(), 0.)
  } else if a.len() >= 3 {
    // FIXME!
    if a.len() > 3 {
      // warn!("Size is {} > 3, pretending it's size 3", a.len());
    }
    Vec3::new(a[0].into_inner(), a[1].into_inner(), a[2].into_inner())
  } else {
    panic!("Invalid size! {:?}", a);
  }
}

// From https://github.com/asny/three-d/blob/master/examples/wireframe/src/main.rs
fn edge_transform(p1: Vec3, p2: Vec3, width: f32) -> Mat4 {
  Mat4::from_translation(p1)
    * Into::<Mat4>::into(Quat::from_arc(
      vec3(1.0, 0.0, 0.0),
      (p2 - p1).normalize(),
      None,
    ))
    * Mat4::from_nonuniform_scale((p1 - p2).magnitude(), width, width)
}

fn picked_id<G: Geometry, M: Material, Ix: IndexType>(
  context: &Context,
  camera: &Camera,
  position: &LogicalPoint,
  geometries: &Gm<G, M>,
  index_to_position: &[(Ix, NotNan1D)],
) -> Option<Ix> {
  if let Some(pick) = pick(context, camera, *position, geometries) {
    if let Some(&(nearest_id, _)) = index_to_position.iter().min_by_key(
      // Find node index with nearest geometry
      |&(_, position)| {
        NotNan::<f32>::new(pick.distance2(Vector3::<f32> {
          x: position[0].to_f32().unwrap(),
          y: position[1].to_f32().unwrap(),
          z: position[2].to_f32().unwrap(),
        }))
        .unwrap()
      },
    ) {
      return Some(nearest_id);
    }
  }
  None
}

fn draw_axes(context: &Context) -> Axes {
  Axes::new(context, 0.02, 0.2)
}

fn create_node_transform(position: &NotNan1D) -> Mat4 {
  Mat4::from_translation(from_not_nan_1d(position)) * Mat4::from_scale(SPHERE_SIZE)
}

fn color_transform(x: f64) -> f64 {
  f64::powi(x, COLOR_TRANSFORM_POWER)
}

/**
 * Must be created only once
 */
struct WindowManager {
  event_loop: winit::event_loop::EventLoop<()>, // must be created only once
}

impl WindowManager {
  pub fn new() -> Self {
    WindowManager {
      event_loop: winit::event_loop::EventLoop::new(),
    }
  }
  fn setup_window_context(&self, title: &str) -> (winit::window::Window, WindowedContext) {
    let window_builder = winit::window::WindowBuilder::new()
      .with_title(title.to_string())
      .with_min_inner_size(winit::dpi::LogicalSize::new(480, 480))
      .with_inner_size(winit::dpi::LogicalSize::new(960, 1080))
      .with_position(winit::dpi::LogicalPosition::new(0, 0));

    let window = window_builder.build(&self.event_loop).unwrap();
    let context = WindowedContext::from_winit_window(
      &window,
      three_d::SurfaceSettings {
        vsync: false, // Wayland hangs in swap_buffers when one window is minimized or occluded
        ..three_d::SurfaceSettings::default()
      },
    )
    .unwrap();

    (window, context)
  }
}

pub struct InstancedMeshManager {
  camera: Camera,
  control: OrbitControl,
  index_to_position: Vec<(usize, NotNan1D)>, // Not HashMap since multiple positions can have the same index
  node_instances: Instances,
  edge_instances: Instances,
}

impl Default for InstancedMeshManager {
  fn default() -> Self {
    Self::new()
  }
}

impl InstancedMeshManager {
  pub fn new() -> Self {
    let camera = Camera::new_perspective(
      Viewport::new_at_origo(1, 1),
      vec3(10.0, 10.0, 10.0),
      vec3(0.0, 0.0, 0.0),
      vec3(0.0, 0.0, 1.0),
      degrees(45.0),
      0.001,
      1000.0,
    );
    let control = OrbitControl::new(*camera.target(), 1.0, 100.0);
    InstancedMeshManager {
      camera,
      control,
      index_to_position: Vec::new(),
      node_instances: Instances {
        colors: Some(Vec::<Srgba>::new()),
        ..Default::default()
      },
      edge_instances: Instances {
        colors: Some(Vec::<Srgba>::new()),
        ..Default::default()
      },
    }
  }

  fn add_node(
    &mut self,
    index: Option<usize>,
    position: &NotNan1D,
    (r, g, b, a): (u8, u8, u8, u8),
  ) {
    let transform = create_node_transform(position);
    self.node_instances.transformations.push(transform);
    self
      .node_instances
      .colors
      .as_mut()
      .unwrap()
      .push(Srgba::new(r, g, b, a));
    if let Some(index) = index {
      self.index_to_position.push((index, position.clone()));
    }
  }

  fn add_edge(&mut self, p1: &NotNan1D, p2: &NotNan1D, (r, g, b, a): (u8, u8, u8, u8), width: f32) {
    let transform = edge_transform(from_not_nan_1d(p1), from_not_nan_1d(p2), width);
    self.edge_instances.transformations.push(transform);
    self
      .edge_instances
      .colors
      .as_mut()
      .unwrap()
      .push(Srgba::new(r, g, b, a));
  }

  fn add_path(&mut self, path: &Vec<GraphNode>) {
    for node in path {
      self.add_node(None, node, (0, 200, 0, 255));
      if node.len() >= 6 {
        self.add_edge(
          &node.slice(s![0..3]).to_owned(),
          &node.slice(s![3..6]).to_owned(),
          (100, 100, 0, 255),
          0.005,
        );
      }
    }
  }

  pub fn clear(&mut self) {
    self.node_instances.transformations.clear();
    self.node_instances.colors.as_mut().unwrap().clear();
    self.edge_instances.transformations.clear();
    self.edge_instances.colors.as_mut().unwrap().clear();
    self.index_to_position.clear();
  }

  fn draw_3dof_arm_position(
    &mut self,
    index: Option<usize>,
    position: &NotNan1D,
    color: [u8; 3],
    alpha: u8,
  ) {
    let (elbow_position, eef_position, base_position) = unpack_position(position);
    self.add_node(
      index,
      &(base_position + &to_not_nan_1d(&[0., 0., eef_position[0].into()])),
      (color[0], color[1], color[2], alpha),
    );
  }

  fn draw_arm_position(
    &mut self,
    index: Option<usize>,
    position: &NotNan1D,
    elbow_color: [u8; 3],
    eef_color: [u8; 3],
    base_color: [u8; 3],
    alpha: u8,
    edge_width: f32,
  ) {
    let (elbow_position, eef_position, base_position) = unpack_position(position);
    self.add_node(
      index,
      &elbow_position,
      (elbow_color[0], elbow_color[1], elbow_color[2], alpha),
    );
    self.add_node(
      index,
      &eef_position,
      (eef_color[0], eef_color[1], eef_color[2], alpha),
    );
    self.add_node(
      index,
      &base_position,
      (base_color[0], base_color[1], base_color[2], alpha),
    );
    if edge_width > 0.0 {
      self.add_edge(
        &elbow_position,
        &eef_position,
        (100, 100, 0, alpha),
        edge_width,
      );
      self.add_edge(
        &elbow_position,
        &base_position,
        (100, 100, 0, alpha),
        edge_width,
      );
    }
  }

  fn draw_arm_path(
    &mut self,
    goal_index: &NagIndex,
    nag: &Nag,
    graph: &Graph,
    elbow_color: [u8; 3],
    eef_color: [u8; 3],
    base_color: [u8; 3],
    alpha: u8,
    edge_width: f32,
  ) {
    let mut nag_index = Some(goal_index);
    while nag_index.is_some() {
      let current_nag_index = *nag_index.unwrap();
      let node = &nag[current_nag_index];
      let physical_node = &graph[node.physical_index];
      self.draw_arm_position(
        Some(current_nag_index.index()),
        physical_node,
        elbow_color,
        eef_color,
        base_color,
        alpha,
        edge_width,
      );
      nag_index = node.parent.as_ref();
    }
  }
}

/**
 * Must be created only once
 */
pub struct Visualizer {
  window_manager: WindowManager,
  instanced_mesh_manager: InstancedMeshManager,
  obstacle_generators: Vec<Rc<dyn Fn(&WindowedContext) -> Gm<Mesh, PhysicalMaterial>>>,
}

impl Default for Visualizer {
  fn default() -> Self {
    Self::new()
  }
}

impl Visualizer {
  pub fn new() -> Self {
    Visualizer {
      instanced_mesh_manager: InstancedMeshManager::new(),
      window_manager: WindowManager::new(),
      obstacle_generators: Vec::new(),
    }
  }

  pub fn add_sphere_obstacle(&mut self, center: [f32; 3], radius: f32, color: [u8; 3]) {
    self
      .obstacle_generators
      .push(Rc::new(move |context: &WindowedContext| {
        let mut obs = Gm::new(
          Mesh::new(&context, &CpuMesh::sphere(6)),
          PhysicalMaterial::new(
            &context,
            &CpuMaterial {
              albedo: Srgba {
                r: color[0],
                g: color[1],
                b: color[2],
                a: 255,
              },
              ..Default::default()
            },
          ),
        );
        obs.set_transformation(
          Mat4::from_translation(vec3(center[0], center[1], center[2])) * Mat4::from_scale(radius),
        );
        obs
      })
        as Rc<dyn Fn(&WindowedContext) -> Gm<Mesh, PhysicalMaterial>>);
  }

  pub fn add_aabb_obstacle(&mut self, center: [f32; 3], dims: [f32; 3], color: [u8; 3]) {
    self
      .obstacle_generators
      .push(Rc::new(move |context: &WindowedContext| {
        let mut obs = Gm::new(
          Mesh::new(&context, &CpuMesh::cube()),
          PhysicalMaterial::new(
            &context,
            &CpuMaterial {
              albedo: Srgba {
                r: color[0],
                g: color[1],
                b: color[2],
                a: 255,
              },
              ..Default::default()
            },
          ),
        );
        /* Default cube has sides of length 2 */
        obs.set_transformation(
          Mat4::from_translation(vec3(center[0], center[1], center[2]))
            * Mat4::from_nonuniform_scale(dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0),
        );
        obs
      })
        as Rc<dyn Fn(&WindowedContext) -> Gm<Mesh, PhysicalMaterial>>);
  }

  pub fn add_cylinder_obstacle(
    &mut self,
    center: [f32; 3],
    radius_height: [f32; 2],
    color: [u8; 3],
  ) {
    self
      .obstacle_generators
      .push(Rc::new(move |context: &WindowedContext| {
        let mut obs = Gm::new(
          Mesh::new(&context, &CpuMesh::cylinder(12)),
          PhysicalMaterial::new(
            &context,
            &CpuMaterial {
              albedo: Srgba {
                r: color[0],
                g: color[1],
                b: color[2],
                a: 255,
              },
              ..Default::default()
            },
          ),
        );
        let radius = radius_height[0];
        let height = radius_height[1];
        /* Default cylinder is around the x-axis in the range [0..1] and with radius 1 */
        obs.set_transformation(
          Mat4::from_translation(vec3(center[0], center[1], center[2] - height / 2.0))
            * Mat4::from_nonuniform_scale(radius, radius, height)
            * Mat4::from_angle_y(radians(-PI / 2.0)),
        );
        obs
      })
        as Rc<dyn Fn(&WindowedContext) -> Gm<Mesh, PhysicalMaterial>>);
    /* Create cylinder end caps */
    self
      .obstacle_generators
      .push(Rc::new(move |context: &WindowedContext| {
        let mut obs = Gm::new(
          Mesh::new(&context, &CpuMesh::circle(12)),
          PhysicalMaterial::new(
            &context,
            &CpuMaterial {
              albedo: Srgba {
                r: color[0],
                g: color[1],
                b: color[2],
                a: 255,
              },
              ..Default::default()
            },
          ),
        );
        let radius = radius_height[0];
        let height = radius_height[1];
        /* Default cylinder is around the x-axis in the range [0..1] and with radius 1 */
        obs.set_transformation(
          Mat4::from_translation(vec3(center[0], center[1], center[2] + height / 2.0))
            * Mat4::from_nonuniform_scale(radius, radius, 1.0),
        );
        obs
      })
        as Rc<dyn Fn(&WindowedContext) -> Gm<Mesh, PhysicalMaterial>>);
    self
      .obstacle_generators
      .push(Rc::new(move |context: &WindowedContext| {
        let mut obs = Gm::new(
          Mesh::new(&context, &CpuMesh::circle(12)),
          PhysicalMaterial::new(
            &context,
            &CpuMaterial {
              albedo: Srgba {
                r: color[0],
                g: color[1],
                b: color[2],
                a: 255,
              },
              ..Default::default()
            },
          ),
        );
        let radius = radius_height[0];
        let height = radius_height[1];
        /* Default cylinder is around the x-axis in the range [0..1] and with radius 1 */
        obs.set_transformation(
          Mat4::from_translation(vec3(center[0], center[1], center[2] - height / 2.0))
            * Mat4::from_nonuniform_scale(radius, radius, 1.0),
        );
        obs
      })
        as Rc<dyn Fn(&WindowedContext) -> Gm<Mesh, PhysicalMaterial>>);
  }
  fn draw(
    &mut self,
    mut redraw: impl FnMut(&mut InstancedMeshManager, PhysicalIndex, NagIndex),
    mut keyboard_event_handler: impl FnMut(&mut Key, &mut bool, &mut bool),
  ) {
    let (window, context) = self.window_manager.setup_window_context("Graph");

    let additional_meshes = self
      .obstacle_generators
      .iter()
      .map(|obstacle_generator| obstacle_generator(&context))
      .collect_vec();
    // for obstacle_generator in &self.obstacle_generators {
    //   obstacle_generator(&context);
    // }

    let mut instanced_spheres = Gm::new(
      InstancedMesh::new(&context, &Instances::default(), &CpuMesh::sphere(4)),
      PhysicalMaterial::new_transparent(
        &context,
        &CpuMaterial {
          albedo: Srgba {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
          },
          // Instanced meshes don't work with emissive
          // emissive: Srgba {
          //   r: 255,
          //   g: 255,
          //   b: 255,
          //   a: 255,
          // },
          ..Default::default()
        },
      ),
    );

    let mut instanced_cylinders = Gm::new(
      InstancedMesh::new(&context, &Instances::default(), &CpuMesh::cylinder(3)),
      PhysicalMaterial::new_transparent(
        &context,
        &CpuMaterial {
          albedo: Srgba {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
          },
          // emissive: Srgba {
          //   r: 128,
          //   g: 128,
          //   b: 128,
          //   a: 255,
          // },
          ..Default::default()
        },
      ),
    );

    let light0 = AmbientLight::new(&context, 1.0, Srgba::WHITE);
    let light1 = DirectionalLight::new(&context, 0.1, Srgba::WHITE, &vec3(0.0, 0.5, 0.5)); // Removing this causes error: https://github.com/asny/three-d/issues/404
    let axes = draw_axes(&context);

    let mut frame_input_generator = three_d::FrameInputGenerator::from_winit_window(&window);

    let mut gui = three_d::GUI::new(&context);

    let mut slider_physical_index_input = 0;
    let mut slider_nag_index_input = 0;
    let mut last_slider_physical_index_input = 0;
    let mut last_slider_nag_index_input = 0;
    let mut slider_change_time = Instant::now();
    let mut picker_change_time = Instant::now();
    let mut picker_index = 0;
    let mut is_changed = true;
    self
      .window_manager
      .event_loop
      .run_return(|event, _, control_flow| match &event {
        winit::event::Event::MainEventsCleared => {
          window.request_redraw();
        }
        winit::event::Event::RedrawRequested(_) => {
          let mut frame_input = frame_input_generator.generate(&context);
          let mut panel_height = 0.0;

          gui.update(
            &mut frame_input.events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,
            |gui_context| {
              use three_d::egui::*;
              TopBottomPanel::bottom("bottom_panel").show(gui_context, |ui| {
                use three_d::egui::*;
                ui.heading("Debug Panel");
                ui.add(Slider::new(&mut slider_physical_index_input, 0..=100000).text("PHY"));
                ui.add(Slider::new(&mut slider_nag_index_input, 0..=100000).text("NAG"));
              });
              panel_height = gui_context.used_rect().height();
            },
          );
          let viewport = Viewport {
            x: 0,
            y: 0,
            width: frame_input.viewport.width,
            height: frame_input.viewport.height
              - (panel_height * frame_input.device_pixel_ratio) as u32,
          };
          context.make_current().unwrap();
          self.instanced_mesh_manager.camera.set_viewport(viewport);
          self.instanced_mesh_manager.control.handle_events(
            &mut self.instanced_mesh_manager.camera,
            &mut frame_input.events,
          );

          // FIXME: This is not very elegant...
          for event in &mut frame_input.events {
            match event {
              Event::MouseRelease { position, .. } => {
                if let Some(picked_id) = picked_id(
                  &context,
                  &self.instanced_mesh_manager.camera,
                  position,
                  &instanced_spheres,
                  &self.instanced_mesh_manager.index_to_position,
                ) {
                  picker_index = picked_id;
                  picker_change_time = Instant::now();
                  is_changed = true;
                }
              }
              Event::KeyRelease { kind, handled, .. } => {
                is_changed = true;
                let mut is_exit = false;
                keyboard_event_handler(kind, handled, &mut is_exit);
                if is_exit {
                  control_flow.set_exit();
                  return;
                }
              }
              _ => {}
            }
          }
          if last_slider_physical_index_input != slider_physical_index_input {
            last_slider_physical_index_input = slider_physical_index_input;
            slider_change_time = Instant::now();
            is_changed = true;
          }
          if last_slider_nag_index_input != slider_nag_index_input {
            last_slider_nag_index_input = slider_nag_index_input;
            slider_change_time = Instant::now();
            is_changed = true;
          }
          let now = Instant::now();
          let mut physical_index_input = 0;
          let mut nag_index_input = 0;
          if now - slider_change_time < now - picker_change_time {
            physical_index_input = slider_physical_index_input;
            nag_index_input = slider_nag_index_input;
          } else {
            physical_index_input = picker_index;
            nag_index_input = picker_index;
          }
          if is_changed {
            redraw(
              &mut self.instanced_mesh_manager,
              PhysicalIndex::new(physical_index_input),
              NagIndex::new(nag_index_input),
            );
          }
          is_changed = false;
          instanced_spheres.set_instances(&self.instanced_mesh_manager.node_instances);
          instanced_cylinders.set_instances(&self.instanced_mesh_manager.edge_instances);
          let screen = frame_input.screen();
          screen
            .clear(ClearState::color_and_depth(1.0, 1.0, 1.0, 1.0, 1.0))
            .render(
              &self.instanced_mesh_manager.camera,
              // https://stackoverflow.com/questions/56496044/how-can-i-transform-an-iterator-trait-object-of-concrete-types-into-an-iterator
              // https://stackoverflow.com/questions/63525018/how-to-convert-vector-of-box-to-vector-of-reference
              [&instanced_spheres, &instanced_cylinders]
                .iter()
                .map(|c| c as _)
                .chain(&axes)
                .chain((&additional_meshes).into_iter().map(|c| c as _)),
              &[&light0, &light1],
            );
          screen.write(|| gui.render());
          context.swap_buffers().unwrap();
          control_flow.set_poll();
          window.request_redraw();
        }
        winit::event::Event::WindowEvent { event, .. } => {
          frame_input_generator.handle_winit_window_event(event);
          match event {
            winit::event::WindowEvent::Resized(physical_size) => {
              context.resize(*physical_size);
            }
            winit::event::WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
              context.resize(**new_inner_size);
            }
            winit::event::WindowEvent::CloseRequested => {
              control_flow.set_exit();
            }
            _ => (),
          }
        }
        _ => {}
      });
  }

  pub fn visualize_graph<'a>(
    &mut self,
    graph: Graph,
    highlights: impl IntoIterator<Item = &'a PhysicalIndex> + Clone,
  ) {
    self.instanced_mesh_manager.clear();
    self.draw(
      |instanced_mesh_manager, picked_id, _| {
        instanced_mesh_manager.clear();
        for index in 0..graph.node_count() {
          let index = graph.from_index(index);
          let position: &NotNan1D = &graph[index];
          instanced_mesh_manager.add_node(Some(index.index()), position, (0, 0, 200, 100));
          for neighbor in graph.neighbors(index) {
            instanced_mesh_manager.add_edge(
              position,
              &graph[neighbor],
              (150, 150, 150, DEFAULT_EDGE_ALPHA),
              DEFAULT_EDGE_WIDTH,
            );
          }
        }
        for &highlight in highlights.clone() {
          let position: &NotNan1D = &graph[highlight];
          instanced_mesh_manager.add_node(None, position, (255, 0, 0, 255));
        }
        if picked_id.index() >= graph.node_count() {
          return;
        }
        instanced_mesh_manager.add_node(None, &graph[picked_id], (255, 0, 0, 255));

        for neighbors in graph.neighbors(picked_id) {
          let neighbor_node = &graph[neighbors];
          instanced_mesh_manager.add_node(None, neighbor_node, (255, 255, 0, 255));
          instanced_mesh_manager.add_edge(
            &graph[picked_id],
            neighbor_node,
            (0, 100, 0, 255),
            DEFAULT_EDGE_WIDTH * 2.0,
          );
        }
      },
      |kind, handled, is_exit| {
        if *kind == Key::Q {
          *handled = true; // override the default keyboard handler
          *is_exit = true;
        }
      },
    );
  }

  pub fn visualize_nag(&mut self, nag: Nag, graph: Graph) {
    self.instanced_mesh_manager.clear();
    let grad = colorgrad::viridis();
    let max_value = nag
      .node_references()
      .max_by_key(|&(_, x)| x.value)
      .unwrap()
      .1
      .value;

    self.draw(
      |instanced_mesh_manager, physical_index_input, nag_index_input| {
        instanced_mesh_manager.clear();
        for (nag_index, node) in nag.node_references() {
          let mut augmented_physical_node = graph[node.physical_index].clone();
          if augmented_physical_node.len() < 3 {
            augmented_physical_node
              .append(Axis(0), to_not_nan_1d(&[0.0]).view())
              .unwrap();
          }
          let random_vec = (Array::from_iter(thread_rng().gen::<[f32; 3]>()) - 0.5) * 0.02;
          let visual_value = color_transform((node.value / max_value).to_f64().unwrap());
          let mut color = grad.at(visual_value).to_rgba8();
          // color[3] = min(200, max(f64::round(255. * visual_value) as u8, 50));
          color[3] = 255;
          if nag_index == nag_index_input {
            color = [255, 0, 0, 255];
          }
          if node.physical_index == physical_index_input {
            color = [255, 0, 0, 255];
          }
          // Add random vec to avoid overlapping nodes
          // let position: NotNan1D = <&GraphNode as Into<&NotNan1D>>::into(&augmented_physical_node)
          //   .slice(s![0..3])
          //   .to_owned()
          //   + &random_vec;
          let position: NotNan1D = <&GraphNode as Into<&NotNan1D>>::into(&augmented_physical_node)
            .slice(s![0..3])
            .to_owned();
          instanced_mesh_manager.add_node(Some(nag_index.index()), &position, color.into());
        }
        if nag_index_input.index() > nag.node_count() {
          return;
        }
        let picked_nag_node: &NagNode = &nag[nag_index_input];
        println!("Selected {:?}", picked_nag_node);
        instanced_mesh_manager.add_node(
          None,
          &graph[picked_nag_node.physical_index],
          (255, 255, 255, 255),
        );
        print!("Path Neighborhood: ");
        for neighbor in get_path_neighborhood(&nag, &picked_nag_node) {
          if neighbor == nag_index_input {
            continue;
          }
          print!("{:?}", neighbor);
          let neighbor_node = &nag[neighbor];
          let physical_node: &NotNan1D = &graph[neighbor_node.physical_index];
          instanced_mesh_manager.add_node(None, physical_node, (200, 0, 0, 255));
        }
        println!();

        instanced_mesh_manager.add_path(&get_physical_path(&graph, &nag, &nag_index_input));

        if let Some(parent_id) = picked_nag_node.parent {
          println!("Parent: {:?}", parent_id);
          let parent = &nag[parent_id];
          instanced_mesh_manager.add_node(None, &graph[parent.physical_index], (255, 0, 255, 255));
        }
      },
      |kind, handled, is_exit| {
        if *kind == Key::Q {
          *handled = true; // override the default keyboard handler
          *is_exit = true;
        }
      },
    );
  }

  pub fn visualize_paths(&mut self, graph: &Graph, paths: &Vec<Vec<GraphNode>>) {
    self.instanced_mesh_manager.clear();
    // We need Cell since these variables are mutated in multiple closures
    let path_index = Cell::new(0);
    self.draw(
      |instanced_mesh_manager, _, _| {
        instanced_mesh_manager.clear();
        for index in graph.node_identifiers() {
          let node = &graph[index];
          instanced_mesh_manager.add_node(Some(index.index()), node, (0, 0, 200, 50));

          for neighbor in graph.neighbors(index) {
            instanced_mesh_manager.add_edge(
              node,
              &graph[neighbor],
              (100, 100, 100, DEFAULT_EDGE_ALPHA),
              DEFAULT_EDGE_WIDTH,
            );
          }
        }
        let path = &paths[path_index.get()];
        instanced_mesh_manager.add_path(&path);
        // print_physical_path(path);
      },
      |kind, handled, is_exit| {
        if *kind == Key::N {
          path_index.set(min(paths.len() - 1, path_index.get() + 1));
          *handled = true; // override the default keyboard handler
        }
        if *kind == Key::P {
          path_index.set(path_index.get().saturating_sub(1));
          *handled = true; // override the default keyboard handler
        }
        if *kind == Key::Q {
          *handled = true; // override the default keyboard handler
          *is_exit = true;
        }
      },
    );
  }

  pub fn visualize_goals(&mut self, nag: &Nag, graph: &Graph, nag_goal_indices: &Vec<NagIndex>) {
    let paths = nag_goal_indices
      .iter()
      .map(|goal| get_physical_path(&graph, &nag, goal))
      .collect();
    self.visualize_paths(&graph, &paths)
  }

  pub fn visualize_3d_graph(&mut self, graph: Graph) {
    self.instanced_mesh_manager.clear();
    // We need Cell since these variables are mutated in multiple closures
    self.draw(
      |instanced_mesh_manager, physical_index_input, _| {
        instanced_mesh_manager.clear();
        for index in graph.node_identifiers() {
          instanced_mesh_manager.draw_3dof_arm_position(
            Some(index.index()),
            &graph[index],
            colors::ORANGE,
            255,
          );
        }
      },
      |kind, handled, is_exit| {
        if *kind == Key::Q {
          *handled = true; // override the default keyboard handler
          *is_exit = true;
        }
      },
    );
  }

  pub fn visualize_arm_graph(&mut self, graph: Graph) {
    self.instanced_mesh_manager.clear();
    // We need Cell since these variables are mutated in multiple closures
    let picked_id = Cell::new(None);
    self.draw(
      |instanced_mesh_manager, physical_index_input, _| {
        instanced_mesh_manager.clear();
        if graph.node_count() == 0 {
          return;
        }
        for index in graph.node_identifiers() {
          let random_vec = (Array::from_iter(thread_rng().gen::<[f32; 9]>()) - 0.5) * 0.005;
          instanced_mesh_manager.draw_arm_position(
            Some(index.index()),
            &(&graph[index] + &random_vec),
            colors::ORANGE,
            colors::PURPLE,
            colors::CYAN,
            255,
            0.0,
          );
        }
        picked_id.set(Some(physical_index_input));
        if let Some(picked_id) = picked_id.get() {
          println!("Selected {:?}: {:?}", picked_id, &graph[picked_id]);
          instanced_mesh_manager.draw_arm_position(
            None,
            &graph[picked_id],
            colors::MAGENTA,
            colors::LIME,
            colors::PINK,
            255,
            4.0 * DEFAULT_EDGE_WIDTH,
          );

          print!("Neighbors: ");
          for neighbors in graph.neighbors(picked_id) {
            print!("{:?},", neighbors);
            instanced_mesh_manager.draw_arm_position(
              None,
              &graph[neighbors],
              colors::TEAL,
              colors::LAVENDER,
              colors::BROWN,
              100,
              DEFAULT_EDGE_WIDTH,
            );
          }
          println!();
        }
      },
      |kind, handled, is_exit| {
        if *kind == Key::Q {
          *handled = true; // override the default keyboard handler
          *is_exit = true;
        }
      },
    );
  }

  pub fn visualize_arm_nag(&mut self, nag: Nag, graph: Graph) {
    self.instanced_mesh_manager.clear();
    let grad = colorgrad::viridis();
    let max_value = nag
      .node_references()
      .max_by_key(|&(_, x)| x.value)
      .unwrap()
      .1
      .value;

    // We need Cell since these variables are mutated in multiple closures
    self.draw(
      |instanced_mesh_manager, _, picked_id| {
        instanced_mesh_manager.clear();
        for (nag_index, node) in nag.node_references() {
          let position = &graph[node.physical_index];
          let random_vec = Array::from_iter(
            thread_rng()
              .sample_iter(Uniform::new(-0.005, 0.005))
              .take(position.len()),
          );
          let mut color = grad
            .at((node.value / max_value).to_f64().unwrap())
            .to_rgba8();
          color[3] = min(
            200,
            max(
              f32::round(255. * (node.value / max_value).to_f32().unwrap()) as u8,
              50,
            ),
          );
          instanced_mesh_manager.draw_arm_position(
            Some(nag_index.index()),
            &(position + &random_vec),
            color[0..3].try_into().unwrap(),
            color[0..3].try_into().unwrap(),
            color[0..3].try_into().unwrap(),
            color[3],
            0.0,
          );
        }
        let picked_nag_node: &NagNode = &nag[picked_id];
        println!("Selected {:?}", picked_nag_node);
        instanced_mesh_manager.draw_arm_path(
          &picked_id,
          &nag,
          &graph,
          colors::ORANGE,
          colors::PURPLE,
          colors::CYAN,
          255,
          DEFAULT_EDGE_WIDTH,
        );
        instanced_mesh_manager.draw_arm_position(
          None,
          &graph[picked_nag_node.physical_index],
          colors::MAGENTA,
          colors::LIME,
          colors::PINK,
          255,
          DEFAULT_EDGE_WIDTH * 2.0,
        );
        // for neighbors in nag.neighbors(picked_id) {
        //   let neighbor_node = &nag[neighbors];
        // instanced_mesh_manager.draw_arm_position(&graph[neighbor_node.physical_index], 255, DEFAULT_EDGE_WIDTH);
        // }
      },
      |kind, handled, is_exit| {
        if *kind == Key::Q {
          *handled = true; // override the default keyboard handler
          *is_exit = true;
        }
      },
    );
  }

  pub fn visualize_arm_path(&mut self, nag: &Nag, graph: &Graph, nag_goal_indices: &Vec<NagIndex>) {
    self.instanced_mesh_manager.clear();

    // We need Cell since these variables are mutated in multiple closures
    let path_index = Cell::new(0);

    self.draw(
      |instanced_mesh_manager, _, picked_id| {
        instanced_mesh_manager.clear();

        let goal_index = &nag_goal_indices[path_index.get()];
        instanced_mesh_manager.draw_arm_path(
          goal_index,
          nag,
          graph,
          colors::GREEN,
          colors::BLUE,
          colors::RED,
          255,
          DEFAULT_EDGE_WIDTH,
        );
        // let picked_node: &NagNode = &nag[picked_id];
        // println!("Selected {:?}", picked_node);
        // instanced_mesh_manager.draw_arm_position(
        //   None,
        //   &graph[picked_node.physical_index],
        //   colors::RED,
        //   colors::GREEN,
        //   colors::YELLOW,
        //   255,
        //   4.0 * DEFAULT_EDGE_WIDTH,
        // );

        // print!("Neighbors: ");
        // for neighbors in graph.neighbors(picked_node.physical_index) {
        //   print!("{:?},", neighbors);
        //   instanced_mesh_manager.draw_arm_position(
        //     None,
        //     &graph[neighbors],
        //     colors::TEAL,
        //     colors::LAVENDER,
        //     colors::BROWN,
        //     10,
        //     DEFAULT_EDGE_WIDTH,
        //   );
        // }
        // println!();

        // if let Some(parent_id) = picked_node.parent {
        //   let parent_node = &nag[parent_id];
        //   println!("Parent: {:?}", parent_node);
        //   instanced_mesh_manager.draw_arm_position(
        //     None,
        //     &graph[parent_node.physical_index],
        //     colors::RED,
        //     colors::GREEN,
        //     colors::YELLOW,
        //     255,
        //     2.0 * DEFAULT_EDGE_WIDTH,
        //   );
        // }
      },
      |kind, handled, is_exit| {
        if *kind == Key::N {
          path_index.set(min(nag_goal_indices.len() - 1, path_index.get() + 1));
          *handled = true; // override the default keyboard handler
        }
        if *kind == Key::P {
          path_index.set(path_index.get().saturating_sub(1));
          *handled = true; // override the default keyboard handler
        }
        if *kind == Key::Q {
          *handled = true; // override the default keyboard handler
          *is_exit = true;
        }
      },
    );
  }
}
