import numpy as np
import math
import time

from config import planning_setting

"""
Equivalent to calc_elbow_position in kinematics.rs, calc_elbow_position.py
"""
def calc_elbow_position(forearm_length, upperarm_length, eef, base, is_up):
  """
  >>> np.round(calc_elbow_position(1.0, 0.5, np.array([0.2, 1.0, 0.5]), np.array([0.2, 0.0, 0.0]), True), 5)
  array([0.2, 0. , 0.5])
  """
  l1 = upperarm_length
  A = base
  l2 = forearm_length
  B = eef
  l3 = np.linalg.norm(B - A)
  B_proj = np.array([B[0], B[1], A[2]])
  AB_proj = B_proj - A

  phi1 = np.atan2((B[2] - A[2]), np.linalg.norm(AB_proj))
  if math.isnan(phi1):
    print("phi1 is nan")
    return None
  phi2 = np.acos((l1**2 + l3**2 - l2**2) / (2.0*l1*l3))
  if math.isnan(phi2):
    print("phi2 is nan")
    return None
  if not is_up:
    phi2 = -phi2

  C = A + (AB_proj/np.linalg.norm(AB_proj) * np.cos(phi1 + phi2) + np.array([0, 0, 1]) * np.sin(phi1 + phi2)) * l1
  return C

def create_simple_interpolation_path(l1, l2, xb_start, xb_end, eef_func, num_points, is_up):
  result = []
  for i in np.linspace(0, 1, num_points):
    x_b = (1 - i) * xb_start + i * xb_end
    x_e = eef_func(i)
    x_w = calc_elbow_position(l1, l2, x_e, x_b, is_up)
    result.append([x_w[0], x_w[1], x_w[2], x_e[0], x_e[1], x_e[2], x_b[0], x_b[1], x_b[2]])
  return result

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    start_time = time.time()
    path0 = create_simple_interpolation_path(planning_setting.upperarm_length, planning_setting.forearm_length, planning_setting.initial_x_b, planning_setting.final_x_b, planning_setting.get_eef_position, 200, True)
    print(f"Creating simple interpolation path took {time.time() - start_time} seconds")
    with open("simple_interpolation_path0.txt", "w") as f:
        for row in path0:
            f.write('[' + ",".join([str(x) for x in row]) + "]\n")

    path1 = create_simple_interpolation_path(planning_setting.upperarm_length, planning_setting.forearm_length, planning_setting.initial_x_b, planning_setting.final_x_b, planning_setting.get_eef_position, 200, False)
    with open("simple_interpolation_path1.txt", "w") as f:
        for row in path1:
            f.write('[' + ",".join([str(x) for x in row]) + "]\n")

