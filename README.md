# ROS packages for **Simrad Halo** marine radar
Tools for turning Simrad **`marine_sensor_msgs/RadarSector`** scans into
standard **`sensor_msgs/Image`** Cartesian images—useful for radar odometry algorithm.

---

## Nodes

### `radar_img_crop.cpp`
GPU off-screen converter; accumulates a full 360 ° sweep in polar space,
warps it to Cartesian, and publishes **one image per revolution**.

### `radar_img_range.cpp`
add dynamic range control function node



**Publishes**

Topic: `radar_img_node/radar_image`
Interface: `sensor_msgs/Image`


### `radar_set_state.py`
Change radar state to **standby**

```bash
# Put radar into standby for safety
rosrun radar_converter_node radar_set_state.py _state:=standby
# Kick it back to transmit
rosrun radar_converter_node radar_set_state.py _state:=transmit
```