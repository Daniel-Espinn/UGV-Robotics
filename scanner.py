import pybullet as p
import pybullet_data
import numpy as np
import cv2
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QPushButton, QLabel, QMessageBox, QSizePolicy
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt
import math
import sys
import os
import pyvista as pv
from pyvistaqt import BackgroundPlotter

# Conectar a PyBullet en modo DIRECT
try:
    physics_client = p.connect(p.DIRECT)
    if physics_client < 0:
        raise RuntimeError("Failed to connect to PyBullet physics server")
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
except Exception as e:
    print(f"Error connecting to PyBullet: {e}")
    sys.exit(1)

# Cargar el plano y objetos
try:
    plane_id = p.loadURDF("plane.urdf")
    cube_pos = [2, 2, 0.5]
    cube_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cube_id, basePosition=cube_pos)
    sphere_pos = [-2, -2, 0.5]
    sphere_id = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sphere_id, basePosition=sphere_pos)
    cylinder_pos = [0, 2, 0.5]
    cylinder_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.3, height=1.0)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cylinder_id, basePosition=cylinder_pos)
    robot_pos = [0, 0, 0.5]
    robot_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=0.5)
    robot = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=robot_id, basePosition=robot_pos)
except Exception as e:
    print(f"Error setting up PyBullet environment: {e}")
    if p.isConnected():
        p.disconnect()
    sys.exit(1)

# Configurar la cámara
width = 320
height = 240
fov = 60
aspect = width / height
near = 0.1
far = 10.0
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
camera_height = 0.5
camera_eye = [1, 0, camera_height]  # Cámara hacia adelante en el marco local
camera_target = [2, 0, camera_height]  # Mirar hacia adelante
camera_up = [0, 0, 1]
focal_length = width / (2 * math.tan(math.radians(fov / 2)))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Camera Interface with 3D Point Cloud")
        self.setGeometry(100, 100, 1200, 800)

        # Layout principal
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QGridLayout()
        main_widget.setLayout(layout)

        # Panel de la cámara
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(width, height)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.camera_label, 0, 0)

        # Panel de depth map
        self.depth_label = QLabel()
        self.depth_label.setMinimumSize(width, height)
        self.depth_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.depth_label, 0, 1)

        # Panel del mapa 3D
        try:
            self.plotter = BackgroundPlotter(
                title="3D Point Cloud",
                window_size=(400, 400),
                auto_update=True
            )
            self.plotter.set_background('black')
            self.plotter.camera_position = [(5, 5, 5), (0, 0, 0.5), (0, 0, 1)]
            self.point_cloud_actor = None
            self.axes_added = False
        except Exception as e:
            print(f"Error initializing 3D plotter: {e}")
            QMessageBox.critical(self, "Error", f"Failed to initialize 3D visualization: {e}")
            sys.exit(1)

        # Botón de control
        self.start_stop_button = QPushButton("Start Simulation")
        self.start_stop_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.start_stop_button.clicked.connect(self.toggle_simulation)
        layout.addWidget(self.start_stop_button, 1, 0, 1, 2, Qt.AlignCenter)

        # Inicializar mapas
        self.depth_map = np.zeros((height, width), dtype=np.float32)
        self.object_sizes = {}  # {label: (center_x, center_y, z, radius, shape)}
        self.point_cloud_history = []

        # Configurar simulación
        self.step = 0
        self.steps = 360
        self.angle_step = 2 * math.pi / self.steps
        self.is_running = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)

        # Mostrar imágenes iniciales
        self.update_camera_display(np.zeros((height, width, 3), dtype=np.uint8))
        self.update_depth_display(np.zeros((height, width), dtype=np.float32))

        # Configurar simulación en modo paso a paso
        p.setRealTimeSimulation(0)

    def toggle_simulation(self):
        try:
            if not self.is_running:
                if not p.isConnected():
                    QMessageBox.critical(self, "Error", "PyBullet physics server is not connected.")
                    return
                self.is_running = True
                self.start_stop_button.setText("Stop Simulation")
                self.timer.start(1000 // 30)
            else:
                self.is_running = False
                self.start_stop_button.setText("Start Simulation")
                self.timer.stop()
        except Exception as e:
            print(f"Error in toggle_simulation: {e}")

    def segment_image(self, rgb_image, depth_buffer):
        try:
            rgb_smooth = cv2.GaussianBlur(rgb_image, (5, 5), 0)
            hsv_image = cv2.cvtColor(rgb_smooth, cv2.COLOR_RGB2HSV)
            lower_ground = np.array([0, 0, 100])
            upper_ground = np.array([180, 40, 255])
            ground_mask = cv2.inRange(hsv_image, lower_ground, upper_ground)
            ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            obstacle_mask = cv2.bitwise_not(ground_mask)
            gray = cv2.cvtColor(rgb_smooth, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 80, 150, apertureSize=3)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 150
            obstacle_mask_refined = np.zeros_like(obstacle_mask)
            self.object_sizes.clear()
            for idx, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > min_area:
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        if 0 <= cy < height and 0 <= cx < width:
                            depth = depth_buffer[cy, cx]
                            if depth <= 0 or depth >= 1:
                                continue
                            z = near * far / (far - depth * (far - near))
                            if z > far or z < near:
                                continue
                            depth_window = depth_buffer[max(0, cy-2):min(height, cy+3), max(0, cx-2):min(width, cx+3)]
                            if depth_window.size == 0 or np.std(depth_window) > 0.05:
                                continue
                            X = (cx - width / 2) * z / focal_length
                            Y = (cy - height / 2) * z / focal_length
                            radius_pixels = math.sqrt(area / math.pi)
                            radius_world = radius_pixels * z / focal_length
                            circularity = 4 * math.pi * area / (perimeter * perimeter)
                            shape = 'square' if circularity < 0.8 else 'circle'
                            self.object_sizes[idx] = (X, Y, z, radius_world, shape)
                    cv2.drawContours(obstacle_mask_refined, [contour], -1, 255, -1)
            obstacle_mask = cv2.bitwise_and(obstacle_mask, obstacle_mask_refined)
            ground_mask = cv2.bitwise_not(obstacle_mask)
            segmented_image = rgb_image.copy()
            segmented_image[ground_mask > 0] = [0, 255, 0]
            segmented_image[obstacle_mask > 0] = [255, 0, 0]
            return segmented_image, obstacle_mask
        except Exception as e:
            print(f"Error in image segmentation: {e}")
            return rgb_image, np.zeros(rgb_image.shape[:2], dtype=np.uint8)

    def depth_to_3d(self, x, y, depth):
        if depth <= 0 or depth >= 1 or np.isnan(depth):
            return None
        z = near * far / (far - depth * (far - near))
        if z > far or z < near or np.isnan(z):
            return None
        X = (x - width / 2) * z / focal_length
        Y = (y - height / 2) * z / focal_length
        if np.isnan(X) or np.isnan(Y):
            return None
        return X, Y, z

    def update_simulation(self):
        if not self.is_running or not p.isConnected():
            return
        try:
            yaw = self.step * self.angle_step
            p.resetBasePositionAndOrientation(robot, robot_pos, p.getQuaternionFromEuler([0, 0, yaw]))
            pos, orn = p.getBasePositionAndOrientation(robot)
            rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            camera_eye_world = np.array(pos) + rot_matrix @ np.array(camera_eye)
            camera_target_world = np.array(pos) + rot_matrix @ np.array(camera_target)
            camera_up_world = rot_matrix @ np.array(camera_up)
            _, _, rgb, depth, _ = p.getCameraImage(
                width, height, p.computeViewMatrix(camera_eye_world, camera_target_world, camera_up_world),
                projection_matrix, renderer=p.ER_TINY_RENDERER
            )
            rgb_array = np.reshape(rgb, (height, width, 4))[:, :, :3]
            depth_array = np.reshape(depth, (height, width))
            depth_array = cv2.GaussianBlur(depth_array, (5, 5), 0)
            depth_array = cv2.medianBlur(depth_array, 5)
            rgb_array = np.ascontiguousarray(rgb_array)
            segmented_image, obstacle_mask = self.segment_image(rgb_array, depth_array)
            self.depth_map = near * far / (far - depth_array * (far - near))
            self.depth_map = np.clip(self.depth_map, near, far)
            self.depth_map[np.isnan(self.depth_map)] = far
            display_image = segmented_image.copy()
            display_image = np.ascontiguousarray(display_image)
            self.update_camera_display(display_image)
            self.update_depth_display(self.depth_map)
            self.update_3d_mapping(pos, orn, depth_array, obstacle_mask)
            p.stepSimulation()
            self.step = (self.step + 1) % self.steps
        except Exception as e:
            print(f"Error during simulation update: {e}")
            self.is_running = False
            self.start_stop_button.setText("Start Simulation")
            self.timer.stop()

    def update_camera_display(self, display_image):
        try:
            display_image = np.ascontiguousarray(display_image)
            image = QImage(display_image, width, height, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(image))
        except Exception as e:
            print(f"Error updating camera display: {e}")

    def update_depth_display(self, depth_map):
        try:
            normalized_depth = (depth_map - near) / (far - near)
            normalized_depth = np.clip(normalized_depth, 0, 1)
            depth_image = (normalized_depth * 255).astype(np.uint8)
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
            depth_image = np.ascontiguousarray(depth_image)
            image = QImage(depth_image, width, height, QImage.Format_RGB888)
            self.depth_label.setPixmap(QPixmap.fromImage(image))
        except Exception as e:
            print(f"Error updating depth display: {e}")

    def update_3d_mapping(self, pos, orn, depth_array, obstacle_mask):
        try:
            # Obtener transformación actual del robot
            rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            robot_pos = np.array(pos)
            
            # 1. Generar nube de puntos 3D del frame actual
            points_3d = []
            valid_points = []
            
            # Muestrear la imagen de profundidad
            for y in range(0, height, 2):
                for x in range(0, width, 2):
                    if obstacle_mask[y, x] == 0:
                        continue
                        
                    depth = depth_array[y, x]
                    point_3d = self.depth_to_3d(x, y, depth)
                    if point_3d is None:
                        continue
                        
                    X, Y, Z = point_3d
                    point_camera = np.array([X, Y, Z])
                    point_robot = rot_matrix @ point_camera
                    point_global = point_robot + robot_pos
                    
                    if not np.any(np.isnan(point_global)):
                        # Validar distancia razonable
                        dist = np.linalg.norm(point_global - robot_pos)
                        if 0.1 < dist < far:  # Filtrar puntos muy cercanos o lejanos
                            points_3d.append(point_global)
                            valid_points.append((x, y))

            if points_3d:
                current_points = np.array(points_3d, dtype=np.float32)
                
                # 2. Fusionar con mapa global existente
                if not hasattr(self, 'global_map'):
                    self.global_map = current_points
                    self.global_colors = np.zeros((len(current_points), 3))
                else:
                    # Calcular colores basados en posición (para mejor visualización)
                    colors = np.zeros((len(current_points), 3))
                    colors[:,0] = (current_points[:,0] - current_points[:,0].min()) / (current_points[:,0].max() - current_points[:,0].min())
                    colors[:,1] = (current_points[:,1] - current_points[:,1].min()) / (current_points[:,1].max() - current_points[:,1].min())
                    colors[:,2] = 0.5  # Componente azul fija para mejor contraste
                    
                    self.global_map = np.vstack([self.global_map, current_points])
                    self.global_colors = np.vstack([self.global_colors, colors])
                    
                    # Limitar tamaño para mantener rendimiento
                    if len(self.global_map) > 30000:
                        keep_ratio = 30000 / len(self.global_map)
                        keep_indices = np.random.choice(len(self.global_map), size=30000, replace=False)
                        self.global_map = self.global_map[keep_indices]
                        self.global_colors = self.global_colors[keep_indices]

                # 3. Visualización mejorada
                self.plotter.clear()
                
                # Mostrar nube de puntos global con colores
                if len(self.global_map) > 10:
                    cloud = pv.PolyData(self.global_map)
                    cloud['colors'] = self.global_colors
                    self.plotter.add_mesh(cloud, scalars='colors', rgb=True, point_size=8, 
                                        render_points_as_spheres=True, name='global_map')
                
                # Mostrar objetos del entorno conocidos
                for obj_name, obj in {'cube': cube_pos, 'sphere': sphere_pos, 'cylinder': cylinder_pos}.items():
                    if obj_name == 'cube':
                        mesh = pv.Cube(center=obj, x_length=1.0, y_length=1.0, z_length=1.0)
                    elif obj_name == 'sphere':
                        mesh = pv.Sphere(center=obj, radius=0.5)
                    else:
                        mesh = pv.Cylinder(center=obj, direction=[0,0,1], radius=0.3, height=1.0)
                    self.plotter.add_mesh(mesh, color='red', opacity=0.7, name=obj_name)
                
                # Mostrar robot con orientación
                robot_mesh = pv.Cylinder(center=robot_pos, direction=[0,0,1], radius=0.2, height=0.5)
                self.plotter.add_mesh(robot_mesh, color='green', name='robot')
                
                # Mostrar dirección del robot
                robot_dir = rot_matrix @ np.array([1,0,0])
                arrow = pv.Arrow(start=robot_pos, direction=robot_dir, scale=0.8)
                self.plotter.add_mesh(arrow, color='yellow', name='direction')
                
                # Configuración de vista persistente
                if not hasattr(self, 'view_setup_done'):
                    self.plotter.camera_position = [(8, 8, 5), (0, 0, 0.5), (0, 0, 1)]
                    self.plotter.add_axes(interactive=True)
                    self.plotter.enable_terrain_style()
                    self.plotter.show_grid()
                    self.view_setup_done = True
                    
                self.plotter.render()

        except Exception as e:
            print(f"Error updating 3D mapping: {e}")
            
    def closeEvent(self, event):
        if p.isConnected():
            p.disconnect()
        self.plotter.close()
        event.accept()

# Iniciar la aplicación
if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        app.exec()
    except Exception as e:
        print(f"Error starting application: {e}")
        if p.isConnected():
            p.disconnect()
        sys.exit(1)