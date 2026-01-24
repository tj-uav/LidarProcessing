from pathlib import Path
from rosbags.highlevel import AnyReader
import numpy as np
import pyvista as pv

BAG_PATH = Path("HolybroOut02.bag")
LIDAR_TOPIC = "/ouster/points"

MAX_RANGE = 20.0        # meters
MIN_Z = 0.0             # meters
MAX_Z = 5.0             # meters
MAX_TOTAL_POINTS = 2_000_000

SINGLE_SCAN_ONLY = False

def pointcloud2_to_xyz(msg):
    field_map = {f.name: f.offset for f in msg.fields}

    if not {"x", "y", "z"}.issubset(field_map):
        return np.empty((0, 3), dtype=np.float32)

    step = msg.point_step
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if msg.is_bigendian:
        data = data.byteswap()

    n_points = len(data) // step

    x = np.ndarray(
        shape=(n_points,),
        dtype=np.float32,
        buffer=data,
        offset=field_map["x"],
        strides=(step,)
    )
    y = np.ndarray(
        shape=(n_points,),
        dtype=np.float32,
        buffer=data,
        offset=field_map["y"],
        strides=(step,)
    )
    z = np.ndarray(
        shape=(n_points,),
        dtype=np.float32,
        buffer=data,
        offset=field_map["z"],
        strides=(step,)
    )

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

        ranges = np.linalg.norm(xyz, axis=1)
        mask = (
            (ranges < MAX_RANGE) &
            (xyz[:, 2] >= MIN_Z) &
            (xyz[:, 2] <= MAX_Z)
        )
        xyz = xyz[mask]

        if xyz.size == 0:
            continue

        all_points.append(xyz)
        total_points += xyz.shape[0]

        if SINGLE_SCAN_ONLY:
            break

        if total_points >= MAX_TOTAL_POINTS:
            break


if not all_points:
    raise RuntimeError("No valid points extracted")

points = np.vstack(all_points)

print(f"Points rendered: {points.shape[0]}")
print("X:", points[:, 0].min(), points[:, 0].max())
print("Y:", points[:, 1].min(), points[:, 1].max())
print("Z:", points[:, 2].min(), points[:, 2].max())

cloud = pv.PolyData(points)
cloud["Z"] = points[:, 2]

plotter = pv.Plotter()
plotter.add_mesh(
    cloud,
    scalars="Z",
    cmap="terrain",
    point_size=3,
    render_points_as_spheres=False
)

plotter.add_axes()
plotter.add_title("UAV LiDAR Point Cloud")

plotter.show_bounds(
    grid="front",
    location="outer",
    xtitle="X (m)",
    ytitle="Y (m)",
    ztitle="Z (m)"
)

plotter.reset_camera()
plotter.show()