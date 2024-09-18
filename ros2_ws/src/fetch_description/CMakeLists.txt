cmake_minimum_required(VERSION 3.22)
project(fetch_description)

find_package(ament_cmake REQUIRED)

ament_package()

install(DIRECTORY meshes
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY robots
  DESTINATION share/${PROJECT_NAME}
)

