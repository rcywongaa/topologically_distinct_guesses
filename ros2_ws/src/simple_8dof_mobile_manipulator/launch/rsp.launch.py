from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_rsp_launch


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder(
        "mobile_manipulator", package_name="simple_8dof_mobile_manipulator"
    ).to_moveit_configs()
    return generate_rsp_launch(moveit_config)
