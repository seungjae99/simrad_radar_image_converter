# ROS packages for **Simrad Halo** marine radar
Tools for turning Simrad **`marine_sensor_msgs/RadarSector`** scans into
standard **`sensor_msgs/Image`** Cartesian images—useful for radar odometry algorithm.

---

## Nodes

### `radar_sector_to_image_transmit_node`
GPU off-screen converter; accumulates a full 360 ° sweep in polar space,
warps it to Cartesian, and publishes **one image per revolution**.

| Parameter | Type / Default | 
|-----------|----------------|
| **`input_topic`** | `string` – `/HaloA/data` | 
| **`sensor_range_max`** | `float` – `3000.0` | 
| **`range_limit`** | `float` – `1000.0` | 
| **`pixels_per_meter`** | `double` – `0.08535` |

**Publishes**

| Topic | Type |
|-------|------|
| `~/radar_image` | `sensor_msgs/Image` |


### `radar_set_state.py`
Change radar state to **standby**

```bash
# Put radar into standby for safety
rosrun radar_converter_node radar_set_state.py _state:=standby
# Kick it back to transmit
rosrun radar_converter_node radar_set_state.py _state:=transmit
```