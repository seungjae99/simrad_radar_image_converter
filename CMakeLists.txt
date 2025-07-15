cmake_minimum_required(VERSION 3.0.2)
project(radar_converter_node)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  image_transport
  cv_bridge
  marine_sensor_msgs
  sensor_msgs
)

find_package(Qt5 REQUIRED COMPONENTS Gui OpenGL)
find_package(OpenCV REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp rospy image_transport cv_bridge marine_sensor_msgs sensor_msgs
)


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Qt5Gui_INCLUDE_DIRS}
  ${Qt5OpenGL_INCLUDE_DIRS}
)

add_executable(radar_sector_to_image_transmit_node src/radar_sector_to_image_transmit.cpp)
target_link_libraries(radar_sector_to_image_transmit_node
  Qt5::Gui Qt5::OpenGL
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(test_node src/test.cpp)
target_link_libraries(test_node
  Qt5::Gui Qt5::OpenGL
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)