# LiDAR Point Cloud Visualization from ROS Bag

This project extracts LiDAR point cloud data from a ROS bag file, filters it by range and height, and visualizes it in 3D using Open3D with height-based coloring.

---

## Features

- Extracts points from a ROS bag topic (`/ouster/points`)
- Converts ROS PointCloud2 messages to XYZ coordinates
- Filters points by:
  - Z-axis limits
  - Maximum radial distance
- Visualizes the point cloud with:
  - Height-based coloring
  - Coordinate frame
  - Grid overlay

---

## Requirements

The project is designed to run in a Python environment using **conda**. Required packages:

- `numpy`
- `open3d`
- `laspy`
- `rosbags`
- `pdal`

---