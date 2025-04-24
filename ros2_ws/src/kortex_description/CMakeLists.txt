cmake_minimum_required(VERSION 3.22)

project(kortex_description)


find_package(ament_cmake REQUIRED)
ament_package()

# Install
foreach(dir arms grippers robots launch)
	install(DIRECTORY ${dir}/
		DESTINATION share/${PROJECT_NAME}/${dir})
endforeach(dir)

