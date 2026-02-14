from pathlib import Path
import numpy as np
import json
import pdal
import open3d as o3d
from rosbags.highlevel import AnyReader

BAG_PATH = Path("HolybroOut02.bag")
LIDAR_TOPIC = "/ouster/points"
TEMP_LAS = "temp_raw.las"
FILTERED_LAS = "filtered.las"
MAX_RANGE = 1000.0
MIN_Z = -100.0
MAX_Z = 100.0

def pointcloud2_to_xyz(msg):
    field_map = {f.name: f.offset for f in msg.fields}
    if not {"x", "y", "z"}.issubset(field_map):
        return np.empty((0, 3), dtype=np.float32)
    step = msg.point_step
    data = np.frombuffer(msg.data, dtype=np.uint8)
    n_points = len(data) // step
    x = np.ndarray((n_points,), np.float32, data, field_map["x"], (step,))
    y = np.ndarray((n_points,), np.float32, data, field_map["y"], (step,))
    z = np.ndarray((n_points,), np.float32, data, field_map["z"], (step,))
    xyz = np.column_stack((x, y, z))
    return xyz[np.isfinite(xyz).all(axis=1)]

all_points = []
total_points = 0
with AnyReader([BAG_PATH]) as reader:
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic != LIDAR_TOPIC:
            continue
        msg = reader.deserialize(rawdata, connection.msgtype)
        xyz = pointcloud2_to_xyz(msg)
        if xyz.size == 0:
            continue
        all_points.append(xyz)
        total_points += xyz.shape[0]
        if total_points >= 2_000_000:
            break

points = np.vstack(all_points)

import laspy
header = laspy.LasHeader(point_format=3, version="1.2")
las = laspy.LasData(header)
las.x = points[:, 0]
las.y = points[:, 1]
las.z = points[:, 2]
las.write(TEMP_LAS)

pipeline_dict = {
    "pipeline": [
        TEMP_LAS,
        {
            "type": "filters.range",
            "limits": f"Z[{MIN_Z}:{MAX_Z}]"
        },
        {
            "type": "filters.expression",
            "expression": f"(X*X + Y*Y + Z*Z) <= {MAX_RANGE**2}"
        },
        FILTERED_LAS
    ]
}

pipeline = pdal.Pipeline(json.dumps(pipeline_dict))
pipeline.execute()

filtered = laspy.read(FILTERED_LAS)
points_filtered = np.vstack((filtered.x, filtered.y, filtered.z)).T

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_filtered)
z_vals = points_filtered[:, 2]
z_min = np.percentile(z_vals, 2)
z_max = np.percentile(z_vals, 98)
z_norm = (z_vals - z_min) / (z_max - z_min + 1e-8)
z_norm = np.clip(z_norm, 0, 1)
colors = np.zeros((len(z_norm), 3))
colors[:, 0] = z_norm
colors[:, 2] = 1 - z_norm
pcd.colors = o3d.utility.Vector3dVector(colors)

coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)

def create_grid(size=100, step=1):
    pts = []
    lines = []
    for i in np.arange(-size, size + step, step):
        pts.append([i, -size, 0])
        pts.append([i, size, 0])
        lines.append([len(pts)-2, len(pts)-1])
        pts.append([-size, i, 0])
        pts.append([size, i, 0])
        lines.append([len(pts)-2, len(pts)-1])
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(pts)
    grid.lines = o3d.utility.Vector2iVector(lines)
    grid.paint_uniform_color([0.7, 0.7, 0.7])
    return grid

grid = create_grid()

vis = o3d.visualization.Visualizer()
vis.create_window(width=1200, height=800)
vis.add_geometry(pcd)
vis.add_geometry(coord)
vis.add_geometry(grid)
ctr = vis.get_view_control()
ctr.set_front([0.4, -0.8, -0.4])
ctr.set_lookat([0, 0, 0])
ctr.set_up([0, 0, 1])
ctr.set_zoom(0.8)
opt = vis.get_render_option()
opt.background_color = np.array([1, 1, 1])
opt.point_size = 2.0
vis.run()
vis.destroy_window()