# ROS packages for **Simrad Halo** marine radar
Tools for turning Simrad **`marine_sensor_msgs/RadarSector`** scans into
standard Cartesian images — useful for radar odometry algorithms.

---

## Main Node: `radar_com_img_range`

Accumulates polar `RadarSector` sweeps from the Simrad Halo20+ and publishes
a full 360° Cartesian image once per revolution via offscreen OpenGL rendering.

**Key features**
- Dynamic FBO sizing based on echo count from the first received message
- GPU polar-to-Cartesian projection via GLSL fragment shader
- Publishes `sensor_msgs/CompressedImage` (PNG or JPEG)
- Automatically commands `transmit` on startup and `standby` on shutdown

**Subscribes**

| Topic | Type |
|-------|------|
| `~input_topic` (default: `/HaloA/data`) | `marine_sensor_msgs/RadarSector` |

**Publishes**

| Topic | Type |
|-------|------|
| `~radar_image/compressed` | `sensor_msgs/CompressedImage` |

**Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `~input_topic` | `/HaloA/data` | Incoming RadarSector topic |
| `~compression_format` | `png` | Output codec: `png` or `jpeg` |
| `~jpeg_quality` | `90` | JPEG quality (0–100) |
| `~frame_id` | `map` | TF frame id in published header |

```bash
rosrun radar_converter_node radar_com_img_range \
  _input_topic:=/HaloA/data \
  _compression_format:=jpeg \
  _jpeg_quality:=90
```

---

## Other Nodes

### `radar_img_crop`
GPU off-screen converter with fixed FBO size derived from `sensor_range_max`.
Supports range-limited crop via shader uniform.

### `radar_img_range`
Same pipeline as `radar_com_img_range` but publishes raw `sensor_msgs/Image`
and supports automatic range cycling via `auto_cycle` parameter.

### `radar_set_state.py`
Change radar state manually.

```bash
rosrun radar_converter_node radar_set_state.py _state:=standby
rosrun radar_converter_node radar_set_state.py _state:=transmit
```
