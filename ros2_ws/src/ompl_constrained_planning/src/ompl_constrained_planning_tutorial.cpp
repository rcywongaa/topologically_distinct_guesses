#include <chrono>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <std_msgs/msg/color_rgba.hpp>

#include <fmt/core.h>

const std::chrono::duration TIME_LIMIT = std::chrono::seconds(10);
static const auto LOGGER = rclcpp::get_logger("ompl_constrained_planning_demo");
int main(int argc, char** argv)
{
  using namespace std::chrono_literals;
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  node_options.automatically_declare_parameters_from_overrides(true);
  node_options.allow_undeclared_parameters(true);
  auto node = rclcpp::Node::make_shared("ompl_constrained_planning_demo_node", node_options);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  auto spinner = std::thread([&executor]() { executor.spin(); });

  moveit::planning_interface::MoveGroupInterface move_group_interface(
      node, "mobile_manipulator");
  auto moveit_visual_tools = moveit_visual_tools::MoveItVisualTools{
      node, "world_base", rviz_visual_tools::RVIZ_MARKER_TOPIC,
      move_group_interface.getRobotModel()};

  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

  /* Add ground plane */
  if (false) {
    moveit_msgs::msg::CollisionObject ground_plane;
    ground_plane.header.frame_id = move_group_interface.getPlanningFrame();
    ground_plane.id = "ground_plane";
    shape_msgs::msg::SolidPrimitive primitive;

    // Define the size of the box in meters
    primitive.type = primitive.BOX;
    primitive.dimensions.resize(3);
    primitive.dimensions[primitive.BOX_X] = 10.0;
    primitive.dimensions[primitive.BOX_Y] = 10.0;
    primitive.dimensions[primitive.BOX_Z] = 0.02;

    // Define the pose of the box (relative to the frame_id)
    geometry_msgs::msg::Pose sphere_pose;
    sphere_pose.orientation.w =
        1.0; // We can leave out the x, y, and z components of the
             // quaternion since they are initialized to 0
    sphere_pose.position.x = 0.0;
    sphere_pose.position.y = 0.0;
    sphere_pose.position.z = -0.04;

    ground_plane.primitives.push_back(primitive);
    ground_plane.primitive_poses.push_back(sphere_pose);
    ground_plane.operation = ground_plane.ADD;

    planning_scene_interface.applyCollisionObject(ground_plane);
  }

  /* Add sphere obstacles */
  if (node->has_parameter("sphere_obstacles")) {
    auto obstacle_names =
        node->get_parameter("sphere_obstacles").as_string_array();
    auto radii = node->get_parameter("sphere_obstacle_radii").as_double_array();
    for (size_t i = 0; i < obstacle_names.size(); i++) {
      auto id = obstacle_names[i];
      auto center_x = node->get_parameter("sphere_obstacle_centers." +
                                          std::to_string(i) + ".x")
                          .as_double();
      auto center_y = node->get_parameter("sphere_obstacle_centers." +
                                          std::to_string(i) + ".y")
                          .as_double();
      auto center_z = node->get_parameter("sphere_obstacle_centers." +
                                          std::to_string(i) + ".z")
                          .as_double();
      auto radius = radii[i];

      // Create collision object for the robot to avoid
      auto const collision_object =
          [frame_id = move_group_interface.getPlanningFrame(), id, center_x,
           center_y, center_z, radius] {
            moveit_msgs::msg::CollisionObject collision_object;
            collision_object.header.frame_id = frame_id;
            collision_object.id = id;
            shape_msgs::msg::SolidPrimitive primitive;

            // Define the size of the box in meters
            primitive.type = primitive.SPHERE;
            primitive.dimensions.resize(1);
            primitive.dimensions[primitive.SPHERE_RADIUS] = radius;

            // Define the pose of the box (relative to the frame_id)
            geometry_msgs::msg::Pose sphere_pose;
            sphere_pose.orientation.w =
                1.0; // We can leave out the x, y, and z components of the
                     // quaternion since they are initialized to 0
            sphere_pose.position.x = center_x;
            sphere_pose.position.y = center_y;
            sphere_pose.position.z = center_z;

            collision_object.primitives.push_back(primitive);
            collision_object.primitive_poses.push_back(sphere_pose);
            collision_object.operation = collision_object.ADD;

            return collision_object;
          }();
      // Add the collision object to the scene
      planning_scene_interface.applyCollisionObject(collision_object);
    }
  }

  /* AABB obstacles */
  if (node->has_parameter("aabb_obstacles")) {
    auto obstacle_names =
        node->get_parameter("aabb_obstacles").as_string_array();
    for (size_t i = 0; i < obstacle_names.size(); i++) {
      auto id = obstacle_names[i];
      auto center_x = node->get_parameter("aabb_obstacle_centers." +
                                          std::to_string(i) + ".x")
                          .as_double();
      auto center_y = node->get_parameter("aabb_obstacle_centers." +
                                          std::to_string(i) + ".y")
                          .as_double();
      auto center_z = node->get_parameter("aabb_obstacle_centers." +
                                          std::to_string(i) + ".z")
                          .as_double();
      auto dim_x =
          node->get_parameter("aabb_obstacle_dims." + std::to_string(i) + ".x")
              .as_double();
      auto dim_y =
          node->get_parameter("aabb_obstacle_dims." + std::to_string(i) + ".y")
              .as_double();
      auto dim_z =
          node->get_parameter("aabb_obstacle_dims." + std::to_string(i) + ".z")
              .as_double();

      // Create collision object for the robot to avoid
      auto const collision_object =
          [frame_id = move_group_interface.getPlanningFrame(), id, center_x,
           center_y, center_z, dim_x, dim_y, dim_z] {
            moveit_msgs::msg::CollisionObject collision_object;
            collision_object.header.frame_id = frame_id;
            collision_object.id = id;
            shape_msgs::msg::SolidPrimitive primitive;

            // Define the size of the box in meters
            primitive.type = primitive.BOX;
            primitive.dimensions.resize(3);
            primitive.dimensions[primitive.BOX_X] = dim_x;
            primitive.dimensions[primitive.BOX_Y] = dim_y;
            primitive.dimensions[primitive.BOX_Z] = dim_z;

            // Define the pose of the box (relative to the frame_id)
            geometry_msgs::msg::Pose aabb_pose;
            aabb_pose.orientation.w =
                1.0; // We can leave out the x, y, and z components of the
                     // quaternion since they are initialized to 0
            aabb_pose.position.x = center_x;
            aabb_pose.position.y = center_y;
            aabb_pose.position.z = center_z;

            collision_object.primitives.push_back(primitive);
            collision_object.primitive_poses.push_back(aabb_pose);
            collision_object.operation = collision_object.ADD;

            return collision_object;
          }();
      // Add the collision object to the scene
      planning_scene_interface.applyCollisionObject(collision_object);
    }
  }
  /* Cylinder obstacles */
  if (node->has_parameter("cylinder_obstacles")) {
    auto obstacle_names =
        node->get_parameter("cylinder_obstacles").as_string_array();
    for (size_t i = 0; i < obstacle_names.size(); i++) {
      auto id = obstacle_names[i];
      auto center_x = node->get_parameter("cylinder_obstacle_centers." +
                                          std::to_string(i) + ".x")
                          .as_double();
      auto center_y = node->get_parameter("cylinder_obstacle_centers." +
                                          std::to_string(i) + ".y")
                          .as_double();
      auto center_z = node->get_parameter("cylinder_obstacle_centers." +
                                          std::to_string(i) + ".z")
                          .as_double();
      auto radii =
          node->get_parameter("cylinder_obstacle_radii").as_double_array();
      auto heights =
          node->get_parameter("cylinder_obstacle_heights").as_double_array();

      auto radius = radii[i];
      auto height = heights[i];
      // Create collision object for the robot to avoid
      auto const collision_object =
          [frame_id = move_group_interface.getPlanningFrame(), id, center_x,
           center_y, center_z, radius, height] {
            moveit_msgs::msg::CollisionObject collision_object;
            collision_object.header.frame_id = frame_id;
            collision_object.id = id;
            shape_msgs::msg::SolidPrimitive primitive;

            // Define the size of the box in meters
            primitive.type = primitive.CYLINDER;
            primitive.dimensions.resize(2);
            primitive.dimensions[primitive.CYLINDER_RADIUS] = radius;
            primitive.dimensions[primitive.CYLINDER_HEIGHT] = height;

            // Define the pose of the box (relative to the frame_id)
            geometry_msgs::msg::Pose cylinder_pose;
            cylinder_pose.orientation.w =
                1.0; // We can leave out the x, y, and z components of the
                     // quaternion since they are initialized to 0
            cylinder_pose.position.x = center_x;
            cylinder_pose.position.y = center_y;
            cylinder_pose.position.z = center_z;

            collision_object.primitives.push_back(primitive);
            collision_object.primitive_poses.push_back(cylinder_pose);
            collision_object.operation = collision_object.ADD;

            return collision_object;
          }();
      // Add the collision object to the scene
      planning_scene_interface.applyCollisionObject(collision_object);
    }
  }
  // Resets the demo by cleaning up any constraints and markers
  auto reset_demo = [&move_group_interface, &moveit_visual_tools]() {
    move_group_interface.clearPathConstraints();
    moveit_visual_tools.deleteAllMarkers();
    moveit_visual_tools.trigger();
  };

  std::ofstream planning_time_file("planning_times.txt");

  int trajectory_count = 0;
  // auto start_time = std::chrono::system_clock::now();
  // while (rclcpp::ok() &&
  //        std::chrono::system_clock::now() - start_time < TIME_LIMIT) {
  while (rclcpp::ok()) {
    moveit_visual_tools.loadRemoteControl();
    moveit_visual_tools.prompt(
        "Press 'next' in the RvizVisualToolsGui window to "
        "continue to the linear constraint example");
    reset_demo();

    // Create some helpful lambdas
    /* Since we didn't specify the end effector group in the .srdf file, we
     * have to manually specify the end effector link name here */
    auto current_pose = move_group_interface.getCurrentPose("eef");

    // Creates a pose at a given positional offset from the current pose
    auto get_relative_pose = [current_pose](double x, double y, double z) {
      auto target_pose = current_pose;
      target_pose.pose.position.x += x;
      target_pose.pose.position.y += y;
      target_pose.pose.position.z += z;
      return target_pose;
    };

    // We can also plan along a line. We can use the same pose as last time.
    auto target_pose = get_relative_pose(2.0, 0.0, 0.0);

    moveit_visual_tools.publishSphere(current_pose.pose,
                                      rviz_visual_tools::YELLOW, 0.05);
    moveit_visual_tools.publishSphere(target_pose.pose, rviz_visual_tools::CYAN,
                                      0.05);

    // Building on the previous constraint, we can make it a line, by also
    // reducing the dimension of the box in the x-direction.
    moveit_msgs::msg::PositionConstraint line_constraint;
    line_constraint.header.frame_id =
        move_group_interface.getPoseReferenceFrame();
    line_constraint.link_name = move_group_interface.getEndEffectorLink();
    shape_msgs::msg::SolidPrimitive line;
    line.type = shape_msgs::msg::SolidPrimitive::BOX;
    line.dimensions = {5.0, 0.2, 0.2}; // must be twice the desired distance
    line_constraint.constraint_region.primitives.emplace_back(line);

    geometry_msgs::msg::Pose line_pose;
    line_pose.position.x = current_pose.pose.position.x;
    line_pose.position.y = current_pose.pose.position.y;
    line_pose.position.z = current_pose.pose.position.z;
    line_pose.orientation.x = 0.0;
    line_pose.orientation.y = 0.0;
    line_pose.orientation.z = 0.0;
    line_pose.orientation.w = 1.0;
    line_constraint.constraint_region.primitive_poses.emplace_back(line_pose);
    line_constraint.weight = 1.0;
    moveit_visual_tools.publishCuboid(line_pose, line.dimensions[0],
                                      line.dimensions[1], line.dimensions[2],
                                      rviz_visual_tools::TRANSLUCENT_LIGHT);

    moveit_visual_tools.publishLine(current_pose.pose.position,
                                    target_pose.pose.position,
                                    rviz_visual_tools::BLACK);
    moveit_visual_tools.trigger();

    moveit_msgs::msg::Constraints line_constraints;
    line_constraints.position_constraints.emplace_back(line_constraint);
    line_constraints.name = "use_equality_constraints";
    move_group_interface.setPathConstraints(line_constraints);

    // const moveit::core::JointModelGroup *joint_model_group =
    //     move_group_interface.getCurrentState()->getJointModelGroup(
    //         PLANNING_GROUP);
    // start_state.setFromIK(joint_model_group, initial_pose.pose);
    // move_group_interface.setStartState(start_state);

    // move_group_interface.setPositionTarget(target_pose.pose.position.x,
    //                                        target_pose.pose.position.y,
    //                                        target_pose.pose.position.z);

    /* JointValueTarget hinders ability to generate distinct paths */
    // moveit::core::RobotState
    // end_state(*move_group_interface.getCurrentState());
    // end_state.setVariablePosition("base_x", 1.0);
    // move_group_interface.setJointValueTarget(end_state);
    move_group_interface.setPoseTarget(target_pose);

    move_group_interface.setPlanningTime(300.0);
    move_group_interface.setGoalPositionTolerance(0.01);
    // move_group_interface.setNumPlanningAttempts(5);
    move_group_interface.setPlannerId("KPIECEkConfigDefault");

    auto start_time = std::chrono::system_clock::now();
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group_interface.plan(plan) ==
                    moveit::core::MoveItErrorCode::SUCCESS);
    RCLCPP_INFO(LOGGER, "Plan with line constraint %s",
                success ? "SUCCEEDED" : "FAILED");
    planning_time_file
        << std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::system_clock::now() - start_time)
                              .count())
        << "\n";
    RCLCPP_INFO(LOGGER, "Took %dms",
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now() - start_time)
                    .count());
    planning_time_file.flush();

    if (success) {
      std::ofstream trajectory_file("trajectory" +
                                    std::to_string(trajectory_count) + ".txt");

      robot_model_loader::RobotModelLoader robot_model_loader(node);
      const moveit::core::RobotModelPtr &kinematic_model =
          robot_model_loader.getModel();
      moveit::core::RobotStatePtr robot_state(
          new moveit::core::RobotState(kinematic_model));

      auto trajectory = plan.trajectory_.joint_trajectory;
      for (auto point : trajectory.points) {
        auto positions = point.positions;
        robot_state->setJointGroupPositions("mobile_manipulator", positions);
        auto elbow_position =
            robot_state->getGlobalLinkTransform("elbow").translation();
        auto eef_position =
            robot_state->getGlobalLinkTransform("eef").translation();
        auto base_position =
            robot_state->getGlobalLinkTransform("base").translation();

        std::string line =
            fmt::format("[{}, {}, {}, {}, {}, {}, {}, {}]", elbow_position[0],
                        elbow_position[1], elbow_position[2], eef_position[0],
                        eef_position[1], eef_position[2], base_position[0],
                        base_position[1]);
        trajectory_file << line << "\n";
      }

      trajectory_file.close();
      trajectory_count++;
    }
  }
  RCLCPP_INFO(LOGGER, "Quiting...");
  planning_time_file.close();

  moveit_visual_tools.deleteAllMarkers();
  moveit_visual_tools.trigger();
  move_group_interface.clearPathConstraints();

  rclcpp::shutdown();
  spinner.join();
  return 0;
}
