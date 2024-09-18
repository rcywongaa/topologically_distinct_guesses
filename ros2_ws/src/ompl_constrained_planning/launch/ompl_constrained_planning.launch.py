import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    initial_positions = os.path.join(
        get_package_share_directory("moveit2_tutorials"),
        "config",
        "initial_positions.yaml",
    )

    moveit_config = (
        MoveItConfigsBuilder(
            # "mobile_manipulator", package_name="simple_9dof_mobile_manipulator"
            "mobile_manipulator",
            package_name="simple_8dof_mobile_manipulator",
        )
        .robot_description(
            file_path="config/mobile_manipulator.urdf.xacro",
            mappings={"initial_positions_file": initial_positions},
        )
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    # Set planning pipeline parameters
    moveit_config.planning_pipelines["ompl"]["mobile_manipulator"][
        "enforce_constrained_state_space"
    ] = True
    moveit_config.planning_pipelines["ompl"]["mobile_manipulator"][
        "projection_evaluator"
    ] = "joints(base_x, base_y, shoulder_pan, shoulder_lift, elbow_flex)"

    # Start the actual move_group node/action server
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    planning_setting = os.path.join(
        get_package_share_directory("moveit2_tutorials"),
        "config",
        "planning_setting.yaml",
    )
    # Demo OMPL constrained planning node
    demo_node = Node(
        package="moveit2_tutorials",
        executable="ompl_constrained_planning",
        output="both",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            planning_setting,
        ],
    )

    # RViz
    rviz_config_file = (
        get_package_share_directory("moveit2_tutorials")
        + "/launch/ompl_constrained_planning.rviz"
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
        ],
    )

    # Static TF
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["--frame-id", "world", "--child-frame-id", "world_base"],
    )

    # Publish TF
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description],
    )

    # ros2_control using FakeSystem as hardware
    ros2_controllers_path = os.path.join(
        # get_package_share_directory("simple_9dof_mobile_manipulator"),
        get_package_share_directory("simple_8dof_mobile_manipulator"),
        "config",
        "ros2_controllers.yaml",
    )
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[ros2_controllers_path],
        remappings=[
            ("/controller_manager/robot_description", "/robot_description"),
        ],
        output="both",
    )

    # Load controllers
    load_controllers = []
    for controller in [
        "mobile_manipulator_controller",
        "joint_state_broadcaster",
    ]:
        load_controllers += [
            ExecuteProcess(
                cmd=["ros2 run controller_manager spawner {}".format(controller)],
                shell=True,
                output="screen",
            )
        ]

    return LaunchDescription(
        [
            static_tf,
            robot_state_publisher,
            rviz_node,
            run_move_group_node,
            demo_node,
            ros2_control_node,
        ]
        + load_controllers
    )
