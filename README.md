# ROS packages for Simrad marine radar

This repository provides tools for converting Simrad Halo `RadarSector` messages into standard ROS `sensor_msgs/Image` Cartesian images—useful for radar odometry algorithm.

## Nodes
- **radar_sector_to_image_node**  
  GPU offscreen converter: accumulates full 360° sweep using Qt5/OpenGL and publishes a single Cartesian image per revolution.

## Topic
`/radar_img_node/radar_image`