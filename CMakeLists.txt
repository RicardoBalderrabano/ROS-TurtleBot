cmake_minimum_required(VERSION 3.0.2)
project(balderrabano_rodriguez)


find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  actionlib
  actionlib_msgs
  geometry_msgs
  message_generation
)

catkin_python_setup()

add_action_files(
  FILES
  Controller.action
)

generate_messages(
  DEPENDENCIES
  std_msgs
  actionlib_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS 
    rospy 
    std_msgs 
    sensor_msgs 
    actionlib 
    actionlib_msgs 
    geometry_msgs 
    message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  scripts/global_planner.py
  scripts/local_planner.py
  scripts/goalPublisher.py
  scripts/PID_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY src/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
  USE_SOURCE_PERMISSIONS
)
