# simulation.py
import pybullet as p
import pybullet_data
import time
import numpy as np
import math

class RobotController:
    """
    Controlador avanzado que traduce comandos de alto nivel 
    a comandos específicos de las ruedas del robot con física mejorada
    """
    def __init__(self, robot_id, wheel_joints):
        self.robot_id = robot_id
        self.wheel_joints = wheel_joints
        
        # Configuración del robot (geometría y física)
        self.wheel_radius = 0.12  # Radio de las ruedas en metros
        self.robot_width = 0.5    # Distancia entre ruedas izquierda-derecha
        self.robot_length = 1.0   # Distancia aproximada a lo largo
        
        # Parámetros de control mejorados
        self.max_wheel_velocity = 25.0  # rad/s máximo por rueda
        self.wheel_velocities = [0.0, 0.0]  # L, R (solo las motorizadas)
        
        # Control PID mejorado para cada rueda motorizada
        self.pid_gains = {'kp': 120.0, 'ki': 1.0, 'kd': 3.0}  # Aumentado para más peso
        self.wheel_errors = [0.0, 0.0]
        self.wheel_integral = [0.0, 0.0]
        self.wheel_previous = [0.0, 0.0]
        
        # Límites de aceleración realistas (reducidos para más realismo)
        self.max_acceleration = 15.0  # rad/s²
        self.current_velocities = [0.0, 0.0]
        
        # Filtro de suavizado para comandos
        self.velocity_filter_alpha = 0.8  # Más suavizado para peso
        self.filtered_velocities = [0.0, 0.0]
        
        print("🤖 Robot Controller Mejorado inicializado")
        print(f"   Ruedas motorizadas: {len(wheel_joints)} (L, R)")
        print(f"   Radio rueda: {self.wheel_radius}m")
        print(f"   Dimensiones: {self.robot_length}x{self.robot_width}m")
        print(f"   PID: Kp={self.pid_gains['kp']}, Ki={self.pid_gains['ki']}, Kd={self.pid_gains['kd']}")

    def set_movement_command(self, linear_vel, angular_vel, smooth=True):
        """
        Convierte velocidad lineal y angular del robot a velocidades de ruedas individuales
        con cinemática diferencial mejorada
        
        Args:
            linear_vel: Velocidad lineal deseada en m/s (positivo = adelante)
            angular_vel: Velocidad angular deseada en rad/s (positivo = giro izquierda)
            smooth: Aplicar suavizado a los comandos
        """
        # Cinemática diferencial mejorada para robot diferencial
        half_width = self.robot_width / 2.0
        
        # Calcular velocidades lineales para lados izquierdo y derecho
        left_wheel_linear_vel = linear_vel - (angular_vel * half_width)
        right_wheel_linear_vel = linear_vel + (angular_vel * half_width)
        
        # Convertir a velocidades angulares
        target_wheel_velocities = [
            left_wheel_linear_vel / self.wheel_radius,   # Izquierda
            right_wheel_linear_vel / self.wheel_radius   # Derecha
        ]
        
        # Aplicar límites de velocidad
        for i in range(2):
            target_wheel_velocities[i] = np.clip(
                target_wheel_velocities[i], 
                -self.max_wheel_velocity, 
                self.max_wheel_velocity
            )
        
        # Aplicar suavizado si está habilitado
        if smooth:
            for i in range(2):
                self.filtered_velocities[i] = (
                    self.velocity_filter_alpha * self.filtered_velocities[i] + 
                    (1.0 - self.velocity_filter_alpha) * target_wheel_velocities[i]
                )
            self.wheel_velocities = self.filtered_velocities.copy()
        else:
            self.wheel_velocities = target_wheel_velocities
        
        # Debug info mejorado
        if abs(linear_vel) > 0.01 or abs(angular_vel) > 0.01:
            print(f"🎯 Comando: lin={linear_vel:.2f}m/s, ang={angular_vel:.2f}rad/s")
            print(f"   Ruedas [L,R]: {[f'{v:.1f}' for v in self.wheel_velocities]} rad/s")

    def apply_wheel_controls(self, dt):
        """
        Aplica control PID avanzado a las ruedas con aceleración limitada
        
        Args:
            dt: Tiempo transcurrido desde la última actualización
        """
        for i in range(2):
            # Obtener velocidad actual de la rueda
            joint_state = p.getJointState(self.robot_id, self.wheel_joints[i])
            current_actual_vel = joint_state[1]
            
            # Limitar aceleración (simular inercia del motor)
            target_vel = self.wheel_velocities[i]
            current_vel = self.current_velocities[i]
            
            vel_diff = target_vel - current_vel
            max_vel_change = self.max_acceleration * dt
            
            if abs(vel_diff) > max_vel_change:
                vel_change = max_vel_change if vel_diff > 0 else -max_vel_change
            else:
                vel_change = vel_diff
                
            self.current_velocities[i] = current_vel + vel_change
            
            # Control PID mejorado
            error = self.current_velocities[i] - current_actual_vel
            self.wheel_integral[i] += error * dt
            # Anti-windup para el término integral
            self.wheel_integral[i] = np.clip(self.wheel_integral[i], -15.0, 15.0)
            
            derivative = (error - self.wheel_errors[i]) / dt if dt > 0 else 0
            self.wheel_errors[i] = error
            
            # Calcular salida PID
            pid_output = (
                self.pid_gains['kp'] * error +
                self.pid_gains['ki'] * self.wheel_integral[i] +
                self.pid_gains['kd'] * derivative
            )
            
            # Calcular fuerza del motor basada en PID (aumentada para más peso)
            motor_force = 250.0 + abs(pid_output) * 80.0  # Fuerza adaptativa aumentada
            motor_force = np.clip(motor_force, 100.0, 500.0)  # Límites aumentados
            
            # Aplicar control de velocidad a la rueda con fuerza adaptativa
            p.setJointMotorControl2(
                self.robot_id,
                self.wheel_joints[i],
                p.VELOCITY_CONTROL,
                targetVelocity=self.current_velocities[i],
                maxVelocity=self.max_wheel_velocity,
                force=motor_force
            )

    def get_wheel_states(self):
        """Obtiene el estado detallado de las ruedas motorizadas"""
        wheel_states = []
        for joint_id in self.wheel_joints:
            joint_state = p.getJointState(self.robot_id, joint_id)
            wheel_states.append({
                'position': joint_state[0],  # Ángulo en radianes
                'velocity': joint_state[1],  # Velocidad angular rad/s
                'reaction_forces': joint_state[2],  # Fuerzas de reacción
                'motor_torque': joint_state[3]     # Torque aplicado
            })
        return wheel_states

    def emergency_stop(self):
        """Detención de emergencia mejorada"""
        print("🛑 PARADA DE EMERGENCIA ACTIVADA")
        self.wheel_velocities = [0.0, 0.0]
        self.current_velocities = [0.0, 0.0]
        self.filtered_velocities = [0.0, 0.0]
        
        # Reset PID
        self.wheel_errors = [0.0, 0.0]
        self.wheel_integral = [0.0, 0.0]
        
        for i in range(2):
            p.setJointMotorControl2(
                self.robot_id,
                self.wheel_joints[i],
                p.VELOCITY_CONTROL,
                targetVelocity=0.0,
                force=400.0  # Fuerza de frenado alta aumentada
            )

class CargoRobotSimulator:
    def __init__(self):
        # Inicializar PyBullet con configuración mejorada
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Configurar la simulación con parámetros más realistas
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)  # 240 Hz para mayor precisión
        p.setRealTimeSimulation(0)  # Simulación paso a paso
        
        # Configuraciones adicionales de simulación
        p.setPhysicsEngineParameter(
            fixedTimeStep=1/240,
            numSolverIterations=15,  # Aumentado para mejor estabilidad
            numSubSteps=2,  # Aumentado para mejor precisión
            contactBreakingThreshold=0.001,
            erp=0.2,
            contactERP=0.2,
            frictionERP=0.2
        )
        
        # Cargar el plano con textura mejorada
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(
            self.plane_id, -1, 
            lateralFriction=1.5,
            spinningFriction=0.2,
            rollingFriction=0.02,
            restitution=0.05
        )
        
        # Crear el robot de carga mejorado
        self.create_cargo_robot()
        
        # Inicializar el controlador mejorado
        self.controller = RobotController(self.robot_id, self.wheel_joints)
        
        # Variables de comando de alto nivel con límites ajustados
        self.target_linear_velocity = 0.0
        self.target_angular_velocity = 0.0  
        self.max_linear_speed = 2.0     
        self.max_angular_speed = 2.0   
        
        # Estado de las luces mejorado
        self.lights_state = {
            'front': False,
            'back': False,
            'left': False,
            'right': False,
            'warning': False
        }
        
        # Configurar la cámara con mejor ángulo
        p.resetDebugVisualizerCamera(
            cameraDistance=5,
            cameraYaw=0,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0]
        )
        
        # Configurar la GUI de PyBullet
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        
        self.simulation_running = False
        self.last_time = time.time()
        
        # IDs de marcadores visuales mejorados
        self.light_markers = {}
        self.create_light_markers()
        
        # Agregar algunos obstáculos para pruebas
        self.create_test_environment()
        
        print("✅ Simulador Mejorado inicializado correctamente")

    def create_cargo_robot(self):
        # === Cuerpo principal (11kg) ===
        body_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.3, 0.12])
        body_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.5, 0.3, 0.12],
            rgbaColor=[0.2, 0.4, 0.8, 1.0]
        )

        # === Ruedas motorizadas (cilindros, 9kg cada una) ===
        active_wheel_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.12, height=0.05)
        active_wheel_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.12,
            length=0.05,
            rgbaColor=[0.3, 0.3, 0.3, 1.0]
        )

        # === Ruedas pasivas (esferas, 1kg cada una) ===
        passive_wheel_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.12)
        passive_wheel_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.12,
            rgbaColor=[0.3, 0.3, 0.3, 1.0]
        )

        # === Posiciones de las ruedas ===
        active_wheel_positions = [
            [0.0, -0.25, -0.12],   # Media izquierda (motorizada)
            [0.0,  0.25, -0.12],    # Media derecha (motorizada)
        ]

        passive_wheel_positions = [
            [ 0.45, -0.25, -0.12],  # Frontal izquierda
            [ 0.45,  0.25, -0.12],  # Frontal derecha
            [-0.45, -0.25, -0.12],  # Trasera izquierda
            [-0.45,  0.25, -0.12],  # Trasera derecha
        ]

        # === Orientaciones ===
        wheel_orientation = p.getQuaternionFromEuler([math.pi/2, 0, 0])  # Cilindros rotados 90° en X
        active_orientations = [wheel_orientation] * 2
        passive_orientations = [[0, 0, 0, 1]] * 4  # Esferas sin orientación

        # === Crear multibody con pesos realistas ===
        # Masas: estructura 11kg, ruedas motorizadas 9kg cada una, ruedas pasivas 1kg cada una
        link_masses = [9.0, 9.0] + [1.0] * 4  # Total: 11 + 9 + 9 + 1 + 1 + 1 + 1 = 33kg
        link_collisions = [active_wheel_collision] * 2 + [passive_wheel_collision] * 4
        link_visuals = [active_wheel_visual] * 2 + [passive_wheel_visual] * 4
        link_positions = active_wheel_positions + passive_wheel_positions
        link_orientations = active_orientations + passive_orientations
        link_inertial_positions = [[0, 0, 0]] * 6
        link_inertial_orientations = [[0, 0, 0, 1]] * 6
        link_parent_indices = [0] * 6
        link_joint_types = [p.JOINT_REVOLUTE] * 2 + [p.JOINT_FIXED] * 4  # Esferas fijas
        link_joint_axes = [[0, 0, 1]] * 2 + [[0, 0, 0]] * 4

        self.robot_id = p.createMultiBody(
            baseMass=11.0,  # Peso de la estructura
            baseCollisionShapeIndex=body_collision,
            baseVisualShapeIndex=body_visual,
            basePosition=[0, 0, 0.24],
            baseOrientation=[0, 0, 0, 1],
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_collisions,
            linkVisualShapeIndices=link_visuals,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=link_inertial_positions,
            linkInertialFrameOrientations=link_inertial_orientations,
            linkParentIndices=link_parent_indices,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axes
        )

        # === Dinámica del cuerpo principal ===
        p.changeDynamics(self.robot_id, -1,
                        lateralFriction=0.5,  # Aumentada
                        spinningFriction=0.2,  # Aumentada
                        rollingFriction=0.1,  # Aumentada
                        restitution=0.05,  # Reducida
                        linearDamping=0.1,  # Aumentada
                        angularDamping=0.2,  # Aumentada
                        mass=11.0)  # Especificar masa explícitamente

        # === Dinámica de las ruedas motorizadas (9kg cada una) ===
        for i in range(2):
            p.changeDynamics(self.robot_id, i,
                            lateralFriction=2.0,  # Aumentada para mejor tracción
                            spinningFriction=0.05,  # Aumentada
                            rollingFriction=0.01,  # Aumentada
                            restitution=0.1,  # Reducida
                            linearDamping=0.2,  # Aumentada
                            angularDamping=0.1,  # Aumentada
                            mass=9.0)  # Especificar masa

        # === Dinámica de las ruedas pasivas (1kg cada una) ===
        for i in range(2, 6):
            p.changeDynamics(self.robot_id, i,
                            lateralFriction=0.8,  # Aumentada para mejor tracción
                            spinningFriction=0.0,  # Aumentada
                            rollingFriction=0.0,  # Aumentada
                            restitution=0.05,  # Reducida
                            linearDamping=0.1,  # Aumentada
                            angularDamping=0.05,  # Aumentada
                            mass=1.0)  # Especificar masa

        # === IDs de las ruedas motorizadas ===
        self.wheel_joints = [0, 1]

        # Desactivar motores de las ruedas pasivas
        for joint_id in self.wheel_joints:
            p.setJointMotorControl2(
                self.robot_id,
                joint_id,
                p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0
            )

        print(f"🚛 Robot REALISTA creado con ID: {self.robot_id}")
        print(f"   Peso total: 33kg (11kg estructura + 18kg ruedas motorizadas + 4kg ruedas pasivas)")
        print(f"   Ruedas: 6 (2 cilíndricas motorizadas de 9kg, 4 esféricas pasivas de 1kg)")

    def create_test_environment(self):
        """Crear un entorno de prueba con algunos obstáculos"""
        # Agregar algunas cajas como obstáculos
        for i in range(3):
            box_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.2, 0.2])
            box_visual = p.createVisualShape(
                p.GEOM_BOX, 
                halfExtents=[0.3, 0.2, 0.2], 
                rgbaColor=[0.8, 0.4, 0.2, 1.0]
            )
            
            box_position = [3 + i * 2, 2 - i, 0.2]
            p.createMultiBody(
                baseMass=5.0,  # Cajas más pesadas
                baseCollisionShapeIndex=box_collision,
                baseVisualShapeIndex=box_visual,
                basePosition=box_position
            )
        
        # Agregar cilindros como obstáculos
        for i in range(2):
            cyl_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=0.4)
            cyl_visual = p.createVisualShape(
                p.GEOM_CYLINDER, 
                radius=0.2, 
                length=0.4,
                rgbaColor=[0.2, 0.8, 0.4, 1.0]
            )
            
            cyl_position = [-3 - i * 2, -2 + i, 0.2]
            p.createMultiBody(
                baseMass=3.0,  # Cilindros más pesados
                baseCollisionShapeIndex=cyl_collision,
                baseVisualShapeIndex=cyl_visual,
                basePosition=cyl_position
            )

    def create_light_markers(self):
        """Crea marcadores visuales mejorados para simular las luces"""
        light_positions = {
            'front': [0.55, 0, 0.15],
            'back': [-0.55, 0, 0.15],
            'left': [0, -0.35, 0.15],
            'right': [0, 0.35, 0.15]
        }
        
        for light_name, pos in light_positions.items():
            # Crear esfera más pequeña y brillante
            visual_id = p.createVisualShape(
                p.GEOM_SPHERE, 
                radius=0.03, 
                rgbaColor=[0.2, 0.2, 0.2, 0.9]
            )
            
            marker_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual_id,
                basePosition=pos
            )
            
            self.light_markers[light_name] = marker_id

    def update_light_markers(self):
        """Actualiza la apariencia de los marcadores de luz con efectos mejorados"""
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
        
        light_offsets = {
            'front': [0.55, 0, -0.09],
            'back': [-0.55, 0, -0.09],
            'left': [0, -0.35, -0.09],
            'right': [0, 0.35, -0.09]
        }
        
        for light_name, offset in light_offsets.items():
            if light_name in self.light_markers:
                # Calcular posición relativa al robot
                world_offset = p.multiplyTransforms(robot_pos, robot_orn, offset, [0, 0, 0, 1])
                new_pos = world_offset[0]
                
                # Colores mejorados según estado
                if self.lights_state[light_name]:
                    if light_name == 'front':
                        color = [1.0, 1.0, 0.9, 1.0]  # Blanco cálido
                    elif light_name == 'back':
                        color = [1.0, 0.1, 0.1, 1.0]  # Rojo intenso
                    else:
                        color = [1.0, 0.6, 0.1, 1.0]  # Naranja brillante
                else:
                    color = [0.2, 0.2, 0.2, 0.5]  # Gris apagado
                
                # Actualizar posición y color
                p.resetBasePositionAndOrientation(self.light_markers[light_name], new_pos, [0, 0, 0, 1])
                p.changeVisualShape(self.light_markers[light_name], -1, rgbaColor=color)

    # Métodos de movimiento mejorados
    def move_forward(self):
        self.target_linear_velocity = self.max_linear_speed
        self.target_angular_velocity = 0.0
        self.lights_state['front'] = True
        self.lights_state['back'] = False
        print("⬆️ Comando: Avanzar")

    def move_backward(self):
        self.target_linear_velocity = -self.max_linear_speed
        self.target_angular_velocity = 0.0
        self.lights_state['front'] = False
        self.lights_state['back'] = True
        print("⬇️ Comando: Retroceder")

    def turn_left(self):
        self.target_linear_velocity = 0.0
        self.target_angular_velocity = self.max_angular_speed
        self.lights_state['left'] = True
        self.lights_state['right'] = False
        print("⬅️ Comando: Girar izquierda")

    def turn_right(self):
        self.target_linear_velocity = 0.0
        self.target_angular_velocity = -self.max_angular_speed
        self.lights_state['left'] = False
        self.lights_state['right'] = True
        print("➡️ Comando: Girar derecha")

    def stop_movement(self):
        self.target_linear_velocity = 0.0
        self.target_angular_velocity = 0.0
        # Apagar luces de dirección
        for light in ['front', 'back', 'left', 'right']:
            if not self.lights_state['warning']:
                self.lights_state[light] = False
        print("⏹️ Comando: Detener")

    def emergency_stop(self):
        self.target_linear_velocity = 0.0
        self.target_angular_velocity = 0.0
        self.controller.emergency_stop()
        # Activar luces de emergencia
        self.lights_state['warning'] = True
        for light in ['left', 'right']:
            self.lights_state[light] = True

    def set_max_speeds(self, linear_speed, angular_speed):
        self.max_linear_speed = max(0.1, min(linear_speed, 3.0))  # Límites más conservadores
        self.max_angular_speed = max(0.1, min(angular_speed, 3.0))

    def set_light_state(self, light_name, state):
        if light_name in self.lights_state and light_name != 'warning':
            self.lights_state[light_name] = state

    def toggle_warning_lights(self):
        self.lights_state['warning'] = not self.lights_state['warning']
        warning_state = self.lights_state['warning']
        for light in ['left', 'right']:
            self.lights_state[light] = warning_state

    def step_simulation(self):
        if self.simulation_running:
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time
            
            # Enviar comandos al controlador con suavizado
            self.controller.set_movement_command(
                self.target_linear_velocity, 
                self.target_angular_velocity,
                smooth=True
            )
            
            # Aplicar controles a las ruedas
            self.controller.apply_wheel_controls(dt)
            
            # Actualizar marcadores de luz
            self.update_light_markers()
            
            # Avanzar simulación
            p.stepSimulation()

    def get_robot_state(self):
        """Obtiene el estado completo del robot con información adicional"""
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
        robot_vel, robot_angular_vel = p.getBaseVelocity(self.robot_id)
        wheel_states = self.controller.get_wheel_states()
        
        # Calcular orientación en Euler
        euler_orn = p.getEulerFromQuaternion(robot_orn)
        
        return {
            'position': robot_pos,
            'orientation': robot_orn,
            'euler_orientation': euler_orn,
            'linear_velocity': robot_vel,
            'angular_velocity': robot_angular_vel,
            'speed': np.linalg.norm(robot_vel),
            'wheels': wheel_states,
            'target_linear_vel': self.target_linear_velocity,
            'target_angular_vel': self.target_angular_velocity,
            'lights': self.lights_state.copy()
        }

    def reset_robot(self):
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0.24], [0, 0, 0, 1])
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])
        self.stop_movement()
        self.controller.emergency_stop()
        
        # Reset luces
        for light in self.lights_state:
            self.lights_state[light] = False
        
        print("🔄 Robot reiniciado")

    def start_simulation(self):
        self.simulation_running = True
        self.last_time = time.time()
        print("▶️ Simulación iniciada")

    def stop_simulation(self):
        self.simulation_running = False
        self.emergency_stop()
        print("⏸️ Simulación detenida")

    def disconnect(self):
        p.disconnect()
        print("👋 PyBullet desconectado")