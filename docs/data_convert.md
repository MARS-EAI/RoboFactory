## Data Format
 
This document outlines the data format used in the Policy-Lightning framework, particularly designed for multi-agent manipulation tasks.

When preparing the dataset, the raw collected data should first be converted into a dense and structured format. The framework currently supports the following three data modalities. During dataset construction, the relevant data fields will be loaded according to the `keys` specified in the `input_meta` configuration.

**Image Data:**

For example, in a two-agent collaborative task with M demonstrations, the image observations and actions used for imitation learning are stored in the following structured format:

```
# N: total number of steps across all M trajectories
/action_0:      (N, 8)           float32    
/action_1:      (N, 8)           float32
/head_cam_0:    (N, 3, H, W)     uint8
/head_cam_1:    (N, 3, H, W)     uint8
/episode_ends:  (M,)             int64       # Inidicate where the episode ends for each demonstration
```

**Point Cloud Data:**

For tasks involving 3D perception using point clouds, the data format is as follows:

```
/action_0:      (N, 8)         float32
/action_1:      (N, 8)         float32
/pointcloud:    (N, p_num, 6)  float32    # p_num = number of points (512 / 1024 for 3D Diffusion Policy); 6 = xyz + RGB
/episode_ends:  (M,)           int64      # Inidicate wthere the episode ends for each demonstration
```

**2D + 3D Fusion Data:**

For tasks involving multi-view 2D and 3D fused observations, the data is structured as:

```
/action_0:      (N, 8)           float32
/action_1:      (N, 8)           float32
/head_cam_0:    (N, 3, H, W)     uint8
/head_cam_1:    (N, 3, H, W)     uint8
/pointcloud:    (N, p_num, 6)    float32    # p_num = number of points (512 / 1024 for 3D Diffusion Policy); 6 = xyz + RGB
/episode_ends:  (M,)             int64      # Inidicate wthere the episode ends for each demonstration
```

******

**Original Data Format:**

The original raw data is generated using the [Robofactory](https://github.com/MARS-EAI/RoboFactory) and stored in an HDF5 format. This includes detailed information on agent states, sensor observations, camera parameters, and actions for each trajectory.

A typical structure looks like the following:

```
# n indicates the number of steps in a single trajectory
/traj_0
/traj_0/obs
/traj_0/obs/agent
/traj_0/obs/agent/panda-0
/traj_0/obs/agent/panda-0/qpos: (n, 9)        float32
/traj_0/obs/agent/panda-0/qvel: (n, 9)        float32
/traj_0/obs/agent/panda-1
/traj_0/obs/agent/panda-1/qpos: (n, 9)        float32
/traj_0/obs/agent/panda-1/qvel: (n, 9)        float32
/traj_0/obs/extra
/traj_0/obs/sensor_param
/traj_0/obs/sensor_param/head_camera_agent0
/traj_0/obs/sensor_param/head_camera_agent0/extrinsic_cv:       (n, 3, 4)     float32
/traj_0/obs/sensor_param/head_camera_agent0/cam2world_gl:       (n, 4, 4)     float32
/traj_0/obs/sensor_param/head_camera_agent0/intrinsic_cv:       (n, 3, 3)     float32
/traj_0/obs/sensor_param/head_camera_agent1
/traj_0/obs/sensor_param/head_camera_agent1/extrinsic_cv:       (n, 3, 4)     float32
/traj_0/obs/sensor_param/head_camera_agent1/cam2world_gl:       (n, 4, 4)     float32
/traj_0/obs/sensor_param/head_camera_agent1/intrinsic_cv:       (n, 3, 3)     float32
/traj_0/obs/sensor_param/head_camera_global
/traj_0/obs/sensor_param/head_camera_global/extrinsic_cv:       (n, 3, 4)     float32
/traj_0/obs/sensor_param/head_camera_global/cam2world_gl:       (n, 4, 4)     float32
/traj_0/obs/sensor_param/head_camera_global/intrinsic_cv:       (n, 3, 3)     float32
/traj_0/obs/sensor_data
/traj_0/obs/sensor_data/head_camera_agent0
/traj_0/obs/sensor_data/head_camera_agent0/depth:       (n, H, W, 1)      int16
/traj_0/obs/sensor_data/head_camera_agent0/rgb:         (n, H, W, 3)      uint8
/traj_0/obs/sensor_data/head_camera_agent1
/traj_0/obs/sensor_data/head_camera_agent1/depth:       (n, H, W, 1)      int16
/traj_0/obs/sensor_data/head_camera_agent1/rgb:         (n, H, W, 3)      uint8
/traj_0/obs/sensor_data/head_camera_global
/traj_0/obs/sensor_data/head_camera_global/depth:       (n, H, W, 1)      int16
/traj_0/obs/sensor_data/head_camera_global/rgb:         (n, H, W, 3)      uint8
/traj_0/actions
/traj_0/actions/panda-0:        (n, 8)        float64
/traj_0/actions/panda-1:        (n, 8)        float64
/traj_0/terminated:     (n,)  bool
/traj_0/truncated:      (n,)  bool
/traj_0/success:        (n,)  bool

/traj_1
...
```