from pathlib import Path
from rosbags.highlevel import AnyReader
import numpy as np
import struct
import open3d as o3d
import matplotlib.pyplot as plt

def pointcloud2_to_xyz(msg):
    fmt = '<fff'
    step = msg.point_step
    data = msg.data
    points = []
    for i in range(0, len(data), step):
        x, y, z = struct.unpack_from(fmt, data, i)
        if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
            points.append((x, y, z))
    return np.array(points)

bagpath = Path('HolybroOut02.bag')
LIDAR_TOPIC = '/ouster/points'
MAX_RANGE = 20.0     # meters
MIN_Z = 0.0          # meters
MAX_Z = 5.0          # meters

all_points = []

with AnyReader([bagpath]) as reader:
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic != LIDAR_TOPIC:
            continue

        msg = reader.deserialize(rawdata, connection.msgtype)
        xyz = pointcloud2_to_xyz(msg)

        ranges = np.linalg.norm(xyz, axis=1)
        xyz = xyz[ranges < MAX_RANGE]
        xyz = xyz[(xyz[:, 2] > MIN_Z) & (xyz[:, 2] < MAX_Z)]

        all_points.append(xyz)

if len(all_points) == 0:
    print("No points found in the bag file.")
    exit()

all_points = np.vstack(all_points)
print(f"Total points after filtering all frames: {all_points.shape[0]}")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points)
o3d.visualization.draw_geometries([pcd])

plt.figure(figsize=(10, 8))
hb = plt.hexbin(all_points[:,0], all_points[:,1], C=all_points[:,2], gridsize=300, cmap='terrain', reduce_C_function=np.mean)
plt.colorbar(hb, label='Height (m)')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.title('UAV LiDAR Point Cloud')
plt.show()