cmake_minimum_required(VERSION 3.22)
project(moveit2_tutorials)

# Common cmake code applied to all moveit packages
find_package(moveit_common REQUIRED)
moveit_package()

set(THIS_PACKAGE_INCLUDE_DEPENDS
  ament_cmake
  Boost
  control_msgs
  Eigen3
  interactive_markers
  moveit_core
  moveit_msgs
  moveit_ros_planning
  moveit_ros_planning_interface
  moveit_servo
  moveit_visual_tools
  pluginlib
  rclcpp
  rclcpp_action
  rviz_visual_tools
  tf2_geometry_msgs
  tf2_geometry_msgs
  tf2_ros
  fmt
)

foreach(Dependency IN ITEMS ${THIS_PACKAGE_INCLUDE_DEPENDS})
  find_package(${Dependency} REQUIRED)
endforeach()

include_directories(${THIS_PACKAGE_INCLUDE_DIRS})

ament_export_dependencies(
  ${THIS_PACKAGE_INCLUDE_DEPENDS}
)

ament_export_include_directories(include)

ament_package()
add_executable(ompl_constrained_planning
    src/ompl_constrained_planning_tutorial.cpp)
target_include_directories(ompl_constrained_planning
    PUBLIC include)
ament_target_dependencies(ompl_constrained_planning
    ${THIS_PACKAGE_INCLUDE_DEPENDS} Boost)

install(TARGETS ompl_constrained_planning
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)
