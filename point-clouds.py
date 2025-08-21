import pybullet as p
import pybullet_data
import numpy as np
import open3d as o3d
import cv2
import math
import time
import torch
from droid import Droid  # Assuming DROID-SLAM is installed (https://github.com/princeton-vl/DROID-SLAM)
import torchvision.transforms as T

# ------------------------------------------------------------------
# 1. Entorno PyBullet
# ------------------------------------------------------------------
def setup_pybullet_environment():
    """Crea una sala 10x10 con dos cubos y plano por defecto."""
    try:
        p.connect(p.GUI)
    except Exception as e:
        print(f"GUI connection failed: {e}. Falling back to DIRECT mode.")
        p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    wall_thickness = 0.1
    wall_height    = 3.0
    room_size      = 10.0

    wall_shape = p.createCollisionShape(p.GEOM_BOX,
                                        halfExtents=[room_size/2, wall_thickness, wall_height/2])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_shape,
                      basePosition=[0,  room_size/2, wall_height/2])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_shape,
                      basePosition=[0, -room_size/2, wall_height/2])

    wall_shape = p.createCollisionShape(p.GEOM_BOX,
                                        halfExtents=[wall_thickness, room_size/2, wall_height/2])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_shape,
                      basePosition=[ room_size/2, 0, wall_height/2])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_shape,
                      basePosition=[-room_size/2, 0, wall_height/2])

    cube_size  = 0.5
    cube_shape = p.createCollisionShape(p.GEOM_BOX,
                                        halfExtents=[cube_size/2]*3)
    cube1 = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=cube_shape,
                              basePosition=[ 2,  2, cube_size/2])
    cube2 = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=cube_shape,
                              basePosition=[-2, -2, cube_size/2])

    p.changeVisualShape(cube1, -1, rgbaColor=[1, 0, 0, 1])
    p.changeVisualShape(cube2, -1, rgbaColor=[0, 0, 1, 1])
    return cube1, cube2

# ------------------------------------------------------------------
# 2. Captura RGB-D
# ------------------------------------------------------------------
def capture_image_and_depth(width=640, height=480, fov=60,
                            near=0.1, far=15.0,
                            camera_pos=[0, 0, 1.5], yaw=0, pitch=-15):
    """Devuelve (rgb, depth) en metros."""
    aspect = width / height
    projection = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    target = [camera_pos[0] + math.cos(math.radians(yaw)),
              camera_pos[1] + math.sin(math.radians(yaw)),
              camera_pos[2] + math.sin(math.radians(pitch))]
    view = p.computeViewMatrix(camera_pos, target, [0, 0, 1])

    _, _, img, depth_buf, _ = p.getCameraImage(width, height,
                                               view, projection,
                                               shadow=True,
                                               renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb  = np.array(img, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
    depth = far * near / (far - (far - near) * np.array(depth_buf))
    return rgb, depth

# ------------------------------------------------------------------
# 3. Nube de puntos
# ------------------------------------------------------------------
def create_point_cloud(rgb, depth, fov, width, height):
    """Crea o3d.geometry.PointCloud en el marco de la cámara."""
    fx = fy = width / (2 * math.tan(math.radians(fov) / 2))
    cx, cy = width / 2, height / 2

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    cols = rgb.reshape(-1, 3) / 255.0
    valid = (pts[:, 2] > 0.1) & (pts[:, 2] < 15.0)
    pts, cols = pts[valid], cols[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd, fx, cx, cy

# ------------------------------------------------------------------
# 4. DROID-SLAM Setup and Processing
# ------------------------------------------------------------------
def initialize_droid_slam(width, height, fov):
    """Inicializa DROID-SLAM con parámetros de cámara."""
    fx = fy = width / (2 * math.tan(math.radians(fov) / 2))
    cx, cy = width / 2, height / 2
    intrinsics = [fx, fy, cx, cy]
    
    # Configuración de DROID-SLAM
    args = type('Args', (), {
        'stereo': False,
        'disable_vis': True,
        'buffer': 512,
        'image_size': [height, width],
        'frontend_window': 16,
        'frontend_nms': 2,
        'frontend_radius': 1,
        'beta': 0.3,
        'frontend_thresh': 15.0,
        'filter_thresh': 2.0,
        'warmup': 8
    })()
    
    droid = Droid(args, intrinsics=intrinsics)
    return droid

def preprocess_image(rgb):
    """Convierte imagen RGB a tensor para DROID-SLAM."""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(rgb).unsqueeze(0).cuda()

# ------------------------------------------------------------------
# 5. Bucle principal
# ------------------------------------------------------------------
def main():
    setup_pybullet_environment()

    width, height = 640, 480
    fov = 60
    steps = 72
    yaw_step = 360 / steps

    # Inicializar DROID-SLAM
    droid = initialize_droid_slam(width, height, fov)
    
    # Pose global y posición de la cámara
    global_pose = np.eye(4)
    camera_pos = np.array([0, 0, 1.5])

    combined = o3d.geometry.PointCloud()

    for i in range(steps):
        yaw = i * yaw_step
        print(f"Frame {i+1}/{steps} yaw={yaw:.1f}°")

        # Captura RGB-D
        rgb, depth = capture_image_and_depth(width=width, height=height,
                                             fov=fov, camera_pos=camera_pos, yaw=yaw)

        # Crear nube de puntos
        pcd, fx, cx, cy = create_point_cloud(rgb, depth, fov, width, height)

        # Procesar con DROID-SLAM
        rgb_tensor = preprocess_image(rgb)
        depth_tensor = torch.from_numpy(depth).float().cuda().unsqueeze(0).unsqueeze(-1)
        timestamp = i / 30.0  # Simular timestamp (30 FPS)

        # Añadir frame a DROID-SLAM
        droid.track(timestamp, rgb_tensor, depth=depth_tensor)

        # Obtener pose estimada
        if i > 0:
            poses = droid.frontend.poses.cpu().numpy()  # Última pose
            if len(poses) > i:
                global_pose = poses[-1]  # Pose en formato 4x4
                camera_pos = (global_pose @ np.array([0, 0, 0, 1]))[:3]  # Actualiza posición
            else:
                print("  → DROID-SLAM tracking failed, skipping pose update")

        # Transformar nube de puntos y combinar
        pcd.transform(global_pose)
        combined += pcd

        p.stepSimulation()
        time.sleep(0.01)

    # Finalizar DROID-SLAM (optimización global y cierre de bucles)
    droid.terminate()

    # Post-procesado
    combined = combined.voxel_down_sample(voxel_size=0.02)
    combined, _ = combined.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    o3d.io.write_point_cloud("final_map.ply", combined)
    o3d.visualization.draw_geometries([combined])
    p.disconnect()

if __name__ == "__main__":
    main()