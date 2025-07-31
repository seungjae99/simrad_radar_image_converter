cmake_minimum_required(VERSION 3.0.2)
project(radar_converter_node)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  image_transport
  cv_bridge
  marine_sensor_msgs
  sensor_msgs
)

find_package(Qt5 REQUIRED COMPONENTS Gui OpenGL)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

catkin_package(
  CATKIN_DEPENDS roscpp rospy image_transport cv_bridge marine_sensor_msgs sensor_msgs
)


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Qt5Gui_INCLUDE_DIRS}
  ${Qt5OpenGL_INCLUDE_DIRS}
)

add_executable(radar_img_crop src/radar_img_crop.cpp)
target_link_libraries(radar_img_crop
  Qt5::Gui Qt5::OpenGL
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(radar_img_range src/radar_img_range.cpp)
target_link_libraries(radar_img_range
  Qt5::Gui Qt5::OpenGL
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# add_executable(test_node src/test.cpp)
# target_link_libraries(test_node
#   Qt5::Gui Qt5::OpenGL
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBRARIES}
# )

add_executable(radar_com_img_range src/radar_com_img_range.cpp)
target_link_libraries(radar_com_img_range
  Qt5::Gui Qt5::OpenGL
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# add_executable(radar_com_img_range2 src/radar_com_img_range2.cpp)
# target_link_libraries(radar_com_img_range2
#   Qt5::Gui Qt5::OpenGL
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBRARIES}
# )