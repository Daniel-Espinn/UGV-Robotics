import pybullet as p
import pybullet_data
import numpy as np
import cv2
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QPushButton, QVBoxLayout, QWidget
from PySide6.QtGui import QPainter, QPen, QBrush, QColor
from PySide6.QtCore import Qt, QPointF
import sys
import heapq

class WorldSimulation:
    def __init__(self):
        self.robot_id = None
        self.plane_id = None
        self.wheel_joints = [0, 1]  # left_wheel_joint, right_wheel_joint
        self.warehouse_scale = 2.0
        self.width, self.height = 640, 480
        self.fov, self.aspect, self.near, self.far = 60, self.width/self.height, 0.1, 100
        self.proj_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)
        self.setup()

    def setup(self):
        try:
            p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            p.setPhysicsEngineParameter(fixedTimeStep=1.0/240.0, numSubSteps=4)
            self.plane_id = p.loadURDF("plane.urdf")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            robot_urdf = os.path.join(script_dir, "simple_robot.urdf")
            if not os.path.exists(robot_urdf):
                print(f"Error: simple_robot.urdf not found in {script_dir}")
                sys.exit(1)
            self.robot_id = p.loadURDF(robot_urdf, [0, 0, 0.2])
            for joint in self.wheel_joints:
                p.setJointMotorControl2(self.robot_id, joint, p.VELOCITY_CONTROL, force=10000, maxVelocity=60.0)
                p.changeDynamics(self.robot_id, joint, lateralFriction=0.5, spinningFriction=0.1, rollingFriction=0.1)
            self.create_warehouse()
        except p.error as e:
            print(f"Simulation setup error: {e}")
            sys.exit(1)

    def create_warehouse(self):
        def create_wall(position, size):
            visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=size, rgbaColor=[0.5, 0.5, 0.5, 1])
            collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=size)
            return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape,
                                    baseVisualShapeIndex=visual_shape, basePosition=position)
        create_wall([0, 8*self.warehouse_scale, 2], [10*self.warehouse_scale, 0.2, 2])
        create_wall([0, -8*self.warehouse_scale, 2], [10*self.warehouse_scale, 0.2, 2])
        create_wall([10*self.warehouse_scale, 0, 2], [0.2, 8*self.warehouse_scale, 2])
        create_wall([-10*self.warehouse_scale, 0, 2], [0.2, 8*self.warehouse_scale, 2])
        for i in range(3):
            create_wall([-5*self.warehouse_scale, -6*self.warehouse_scale + i*6*self.warehouse_scale, 1], [0.5, 2*self.warehouse_scale, 1])
            create_wall([5*self.warehouse_scale, -6*self.warehouse_scale + i*6*self.warehouse_scale, 1], [0.5, 2*self.warehouse_scale, 1])

    def get_robot_state(self):
        try:
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            euler = p.getEulerFromQuaternion(orn)
            lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
            return pos, euler, lin_vel, ang_vel
        except p.error as e:
            print(f"Error getting robot state: {e}")
            return None

    def get_camera_image(self, pos, euler):
        try:
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[pos[0], pos[1], pos[2]+0.3],
                cameraTargetPosition=[pos[0]+np.cos(euler[2]), pos[1]+np.sin(euler[2]), pos[2]],
                cameraUpVector=[0, 0, 1]
            )
            img = p.getCameraImage(self.width, self.height, view_matrix, self.proj_matrix,
                                  shadow=True, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            rgb = np.reshape(img[2], (self.height, self.width, 4))[:,:,:3]
            depth = np.reshape(img[3], (self.height, self.width))
            return rgb, depth
        except p.error as e:
            print(f"Error capturing camera image: {e}")
            return None, None

    def set_wheel_velocities(self, left_speed, right_speed):
        try:
            p.setJointMotorControlArray(self.robot_id, self.wheel_joints, p.VELOCITY_CONTROL,
                                       targetVelocities=[left_speed, right_speed])
            for joint in self.wheel_joints:
                joint_state = p.getJointState(self.robot_id, joint)
                print(f"Joint {joint}: Position={joint_state[0]:.2f}, Velocity={joint_state[1]:.2f}")
        except p.error as e:
            print(f"Error setting wheel velocities: {e}")

    def perform_360_scan(self):
        obstacle_positions = []
        steps = 36  # 10-degree increments
        angular_speed = 2 * np.pi / 2.0  # 360 degrees in 2 seconds
        for i in range(steps):
            self.set_wheel_velocities(-angular_speed, angular_speed)
            for _ in range(int(2.0 / (steps * 1.0/240.0))):  # Simulate for 2/steps seconds
                p.stepSimulation()
            pos, euler, _, _ = self.get_robot_state()
            rgb, depth = self.get_camera_image(pos, euler)
            if rgb is not None and depth is not None:
                temp_obstacles, _ = self.segment_obstacles(rgb, depth, pos, euler)
                obstacle_positions.extend(temp_obstacles)
        self.set_wheel_velocities(0, 0)
        return list(set(tuple(pos) for pos in obstacle_positions))  # Remove duplicates

    def segment_obstacles(self, rgb, depth, robot_pos, robot_euler):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = cv2.inRange(gray, 80, 160)  # Gray walls/shelves
        depth_mask = (depth < self.far) & (depth > self.near)
        mask = cv2.bitwise_and(mask, depth_mask.astype(np.uint8) * 255)
        h, w = mask.shape
        fx = w / (2 * np.tan(np.radians(self.fov/2)))
        cx, cy = w/2, h/2
        obstacle_positions = []
        if np.any(mask):
            y, x = np.where(mask)
            for px, py in zip(x[:10], y[:10]):
                z = depth[py, px]
                if z < self.far:
                    wx = (px - cx) * z / fx
                    wy = (cy - py) * z / fx
                    world_x = robot_pos[0] + wx * np.cos(robot_euler[2]) - wy * np.sin(robot_euler[2])
                    world_y = robot_pos[1] + wx * np.sin(robot_euler[2]) + wy * np.cos(robot_euler[2])
                    obstacle_positions.append([world_x, world_y])
        seg_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for pos in obstacle_positions:
            x, y = self.world_to_scene(pos[0], pos[1], robot_pos, robot_euler)
            seg_img = cv2.circle(seg_img, (int(x % w), int(y % h)), 5, (0, 255, 0), -1)
        return obstacle_positions, seg_img

    def world_to_scene(self, x, y, robot_pos, robot_euler):
        return x, y

    def step_simulation(self):
        try:
            p.stepSimulation()
        except p.error as e:
            print(f"Error in simulation step: {e}")
            return False
        return True

    def disconnect(self):
        try:
            p.disconnect()
        except p.error:
            pass

class RobotController:
    def __init__(self, simulation):
        self.simulation = simulation
        self.target_pos = None
        self.navigate = False
        self.path = []
        self.planned_path = []
        self.scene_width, self.scene_height = 600, 600
        self.world_width, self.world_height = 20, 20
        self.grid_resolution = 0.5  # 0.5m grid for A*
        # Perform initial 360-degree scan
        print("Performing initial 360-degree scan...")
        self.initial_obstacles = self.simulation.perform_360_scan()
        print(f"Detected {len(self.initial_obstacles)} obstacles during initial scan")
        self.app = QApplication(sys.argv)
        self.window = self.MainWindow(self)
        self.window.show()

    class MainWindow(QMainWindow):
        def __init__(self, controller):
            super().__init__()
            self.controller = controller
            self.setWindowTitle("Robot Navigation")
            layout = QVBoxLayout()
            self.map_view = self.MapView(controller)
            self.setMinimumSize(600, 600)
            layout.addWidget(self.map_view)
            self.start_button = QPushButton("Start Navigation")
            self.start_button.clicked.connect(self.controller.toggle_navigation)
            layout.addWidget(self.start_button)
            self.reset_button = QPushButton("Reset Map")
            self.reset_button.clicked.connect(self.controller.reset_map)
            layout.addWidget(self.reset_button)
            container = QWidget()
            container.setLayout(layout)
            self.setCentralWidget(container)

        class MapView(QGraphicsView):
            def __init__(self, controller):
                super().__init__()
                self.controller = controller
                self.scene = QGraphicsScene(0, 0, controller.scene_width, controller.scene_height)
                self.setScene(self.scene)
                self.setRenderHint(QPainter.Antialiasing)
                self.update_map()

            def world_to_scene(self, x, y):
                scene_x = (x + self.controller.world_width/2) * (self.controller.scene_width/self.controller.world_width)
                scene_y = (self.controller.world_height/2 - y) * (self.controller.scene_height/self.controller.world_height)
                return scene_x, scene_y

            def update_map(self):
                self.scene.clear()
                robot_pos, _, _, _ = self.controller.simulation.get_robot_state()
                if len(self.controller.path) == 0 or np.linalg.norm(np.array(robot_pos[:2]) - np.array(self.controller.path[-1])) > 0.05:
                    self.controller.path.append(robot_pos[:2])
                # Draw planned path (gray)
                planned_pen = QPen(QColor(128, 128, 128), 2)
                for i in range(1, len(self.controller.planned_path)):
                    x1, y1 = self.world_to_scene(*self.controller.planned_path[i-1])
                    x2, y2 = self.world_to_scene(*self.controller.planned_path[i])
                    self.scene.addLine(x1, y1, x2, y2, planned_pen)
                # Draw real-time path (blue)
                path_pen = QPen(QColor(0, 0, 255), 2)
                for i in range(1, len(self.controller.path)):
                    x1, y1 = self.world_to_scene(*self.controller.path[i-1])
                    x2, y2 = self.world_to_scene(*self.controller.path[i])
                    self.scene.addLine(x1, y1, x2, y2, path_pen)
                # Draw robot
                robot_x, robot_y = self.world_to_scene(*robot_pos[:2])
                self.scene.addEllipse(robot_x-5, robot_y-5, 10, 10, QPen(Qt.red), QBrush(Qt.red))
                # Draw target
                if self.controller.target_pos:
                    target_x, target_y = self.world_to_scene(*self.controller.target_pos)
                    self.scene.addEllipse(target_x-5, target_y-5, 10, 10, QPen(Qt.green), QBrush(Qt.green))
                # Draw initial obstacles
                obstacle_pen = QPen(QColor(255, 0, 0), 1)
                obstacle_brush = QBrush(QColor(255, 0, 0))
                for obs in self.controller.initial_obstacles:
                    obs_x, obs_y = self.world_to_scene(obs[0], obs[1])
                    self.scene.addEllipse(obs_x-3, obs_y-3, 6, 6, obstacle_pen, obstacle_brush)

            def mousePressEvent(self, event):
                if event.button() == Qt.LeftButton:
                    pos = self.mapToScene(event.position().toPoint())
                    self.controller.target_pos = [
                        (pos.x()/self.controller.scene_width) * self.controller.world_width - self.controller.world_width/2,
                        (self.controller.world_height/2 - (pos.y()/self.controller.scene_height)) * self.controller.world_height
                    ]
                    print(f"Target set to: ({self.controller.target_pos[0]:.2f}, {self.controller.target_pos[1]:.2f})")
                    self.update_map()

    def toggle_navigation(self):
        if not self.navigate and self.target_pos:
            # Use initial obstacles for path planning
            print("Planning path with initial scan data...")
            self.planned_path = self.plan_path(self.initial_obstacles)
            print(f"Planned path with {len(self.planned_path)} waypoints")
        self.navigate = not self.navigate
        self.window.start_button.setText("Stop Navigation" if self.navigate else "Start Navigation")
        print(f"Navigation {'enabled' if self.navigate else 'disabled'}")

    def reset_map(self):
        self.path = []
        self.planned_path = []
        self.window.map_view.update_map()

    def plan_path(self, obstacle_positions):
        grid_width = int(self.world_width / self.grid_resolution)
        grid_height = int(self.world_height / self.grid_resolution)
        grid = np.zeros((grid_height, grid_width), dtype=bool)
        for obs in obstacle_positions:
            gx = int((obs[0] + self.world_width/2) / self.grid_resolution)
            gy = int((self.world_height/2 - obs[1]) / self.grid_resolution)
            if 0 <= gx < grid_width and 0 <= gy < grid_height:
                grid[gy, gx] = True
        start = (int((0 + self.world_width/2) / self.grid_resolution), int((self.world_height/2 - 0) / self.grid_resolution))
        target = (int((self.target_pos[0] + self.world_width/2) / self.grid_resolution), int((self.world_height/2 - self.target_pos[1]) / self.grid_resolution))
        if not (0 <= target[0] < grid_width and 0 <= target[1] < grid_height):
            print("Target outside grid, returning direct path")
            return [[0, 0], self.target_pos]
        def heuristic(a, b):
            return np.hypot(a[0] - b[0], a[1] - b[1])
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, target)}
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == target:
                path = []
                while current in came_from:
                    path.append([((current[0] + 0.5) * self.grid_resolution - self.world_width/2),
                                 (self.world_height/2 - (current[1] + 0.5) * self.grid_resolution)])
                    current = came_from[current]
                path.append([0, 0])
                return path[::-1]
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < grid_width and 0 <= neighbor[1] < grid_height) or grid[neighbor[1], neighbor[0]]:
                    continue
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, target)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        print("No path found, returning direct path")
        return [[0, 0], self.target_pos]

    def navigate_to_target(self):
        if not self.target_pos or not self.planned_path:
            return False
        robot_pos, euler, lin_vel, ang_vel = self.simulation.get_robot_state()
        if robot_pos is None:
            return False
        distances = [np.hypot(wp[0] - robot_pos[0], wp[1] - robot_pos[1]) for wp in self.planned_path]
        waypoint_idx = min(range(len(distances)), key=distances.__getitem__)
        if distances[waypoint_idx] < 0.1 and waypoint_idx < len(self.planned_path) - 1:
            waypoint_idx += 1
        target_x, target_y = self.planned_path[waypoint_idx]
        dx = target_x - robot_pos[0]
        dy = target_y - robot_pos[1]
        distance = np.hypot(dx, dy)
        print(f"Robot at ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), Waypoint at ({target_x:.2f}, {target_y:.2f}), Distance: {distance:.2f}")
        print(f"Linear velocity: ({lin_vel[0]:.2f}, {lin_vel[1]:.2f}), Angular velocity: {ang_vel[2]:.2f}")
        if distance > 0.1 or waypoint_idx < len(self.planned_path) - 1:
            rgb, depth = self.simulation.get_camera_image(robot_pos, euler)
            _, seg_img = self.simulation.segment_obstacles(rgb, depth, robot_pos, euler)
            cv2.imshow("Segmentation View", cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
            obstacle_detected = False
            ray_from = [robot_pos[0], robot_pos[1], 0.3]
            ray_to = [robot_pos[0] + np.cos(euler[2]) * 1.0, robot_pos[1] + np.sin(euler[2]) * 1.0, 0.3]
            ray_result = p.rayTest(ray_from, ray_to)
            if ray_result[0][0] != -1 and ray_result[0][0] != self.simulation.robot_id:
                hit_pos = ray_result[0][3]
                hit_distance = np.sqrt((hit_pos[0] - ray_from[0])**2 + (hit_pos[1] - ray_from[1])**2)
                print(f"Raycast hit at distance: {hit_distance:.2f}")
                if hit_distance < 0.5:
                    obstacle_detected = True
            target_angle = np.arctan2(dy, dx)
            angle_error = np.arctan2(np.sin(target_angle - euler[2]), np.cos(target_angle - euler[2]))
            print(f"Yaw: {euler[2]:.2f}, Angle to waypoint: {target_angle:.2f}, Angle error: {angle_error:.2f}")
            if abs(angle_error) < 0.1:
                angle_error = 0
            base_speed = max(5.0, min(30.0, distance * 5.0)) if not obstacle_detected else 0.0
            turn_speed = angle_error * 0.5
            left_speed = max(0, base_speed + turn_speed)
            right_speed = max(0, base_speed - turn_speed)
            left_speed = min(30.0, left_speed)
            right_speed = min(30.0, right_speed)
            print(f"Setting velocities: Left={left_speed:.2f}, Right={right_speed:.2f}")
            self.simulation.set_wheel_velocities(left_speed, right_speed)
            return False
        else:
            print("Target reached, stopping")
            self.simulation.set_wheel_velocities(0, 0)
            return True
        return False

    def run(self):
        running = True
        while running and p.isConnected():
            self.app.processEvents()
            if not self.window.isVisible():
                print("GUI window closed, exiting...")
                running = False
            if self.navigate:
                target_reached = self.navigate_to_target()
                if target_reached:
                    self.navigate = False
                    self.window.start_button.setText("Start Navigation")
            self.window.map_view.update_map()
            if not self.simulation.step_simulation():
                running = False
        cv2.destroyAllWindows()
        self.simulation.disconnect()
        sys.exit(self.app.exec())

if __name__ == "__main__":
    sim = WorldSimulation()
    controller = RobotController(sim)
    controller.run()