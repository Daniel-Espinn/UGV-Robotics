# robot_interface.py
import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QSlider, QLabel, 
                               QGroupBox, QSpinBox, QDoubleSpinBox, QGridLayout,
                               QCheckBox, QFrame, QProgressBar, QSizePolicy, QScrollArea)
from PySide6.QtCore import QTimer, Qt, QSize
from PySide6.QtGui import QFont, QKeySequence, QShortcut, QPalette, QColor, QIcon

# Importar la simulación
from simulation import CargoRobotSimulator

class CargoRobotInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.simulator = CargoRobotSimulator()
        self.init_ui()
        self.setup_keyboard_shortcuts()
        
        # Timer para la simulación
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.update_simulation)
        self.sim_timer.start(16)  # ~60 FPS
        
        # Estados de teclas presionadas
        self.keys_pressed = {'w': False, 'a': False, 's': False, 'd': False}

    def init_ui(self):
        self.setWindowTitle("🚛 Robot de Carga - Control Realista con Ruedas")
        self.setGeometry(100, 100, 1200, 800)
        
        # Aplicar tema oscuro
        self.apply_dark_theme()
        
        # Widget central con scroll area para responsive
        self.central_widget = QWidget()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.central_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.setCentralWidget(self.scroll_area)
        
        # Layout principal
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(15)
        
        # Panel de control izquierdo
        self.control_panel = self.create_control_panel()
        self.main_layout.addWidget(self.control_panel, stretch=1)
        
        # Panel de información derecho
        self.info_panel = self.create_info_panel()
        self.main_layout.addWidget(self.info_panel, stretch=1)
        
        # Ajustar políticas de tamaño para responsive
        self.control_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.info_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    def apply_dark_theme(self):
        """Aplicar tema oscuro a la aplicación"""
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(45, 45, 55))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(35, 35, 45))
        dark_palette.setColor(QPalette.AlternateBase, QColor(45, 45, 55))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(60, 60, 70))
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(65, 65, 75))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 120, 120))
        dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(120, 120, 120))
        
        self.setPalette(dark_palette)
        
        # Estilo general para la aplicación
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2d2d37;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3a3a45;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 12px;
                background-color: #2d2d37;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #bbbbff;
            }
            QProgressBar {
                border: 2px solid #3a3a45;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                background-color: #25252d;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
            QLabel {
                color: #dddddd;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QCheckBox {
                color: #dddddd;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 2px solid #5a5a65;
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
                border: 2px solid #4CAF50;
            }
            QCheckBox::indicator:unchecked {
                background-color: #353540;
            }
            QSlider::groove:horizontal {
                border: 1px solid #3a3a45;
                height: 8px;
                background: #353540;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #5a5a8a;
                border: 1px solid #5a5a65;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #6a6a9a;
                border: 1px solid #6a6a75;
            }
            QSlider::sub-page:horizontal {
                background: #4CAF50;
                border-radius: 4px;
            }
        """)

    def create_control_panel(self):
        panel = QGroupBox("🎮 Control del Robot de Carga")
        panel.setMinimumWidth(400)
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # Botones de control de simulación
        sim_controls = QGroupBox("⚙️ Control de Simulación")
        sim_layout = QHBoxLayout(sim_controls)
        sim_layout.setContentsMargins(8, 15, 8, 8)
        
        self.start_btn = QPushButton("▶ Iniciar")
        self.start_btn.clicked.connect(self.start_simulation)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover { 
                background-color: #45a049; 
            }
            QPushButton:pressed { 
                background-color: #3d8b40; 
            }
        """)
        
        self.stop_btn = QPushButton("⏸ Detener")
        self.stop_btn.clicked.connect(self.stop_simulation)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover { 
                background-color: #da190b; 
            }
            QPushButton:pressed { 
                background-color: #bd0a00; 
            }
        """)
        
        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.clicked.connect(self.reset_robot)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover { 
                background-color: #1976D2; 
            }
            QPushButton:pressed { 
                background-color: #0d5cb6; 
            }
        """)
        
        self.emergency_btn = QPushButton("🛑 EMERGENCIA")
        self.emergency_btn.clicked.connect(self.emergency_stop)
        self.emergency_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF5722;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 12px;
            }
            QPushButton:hover { 
                background-color: #E64A19; 
            }
            QPushButton:pressed { 
                background-color: #D32F2F; 
            }
        """)
        
        # Hacer botones responsivos
        for btn in [self.start_btn, self.stop_btn, self.reset_btn, self.emergency_btn]:
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            btn.setMinimumHeight(40)
        
        sim_layout.addWidget(self.start_btn)
        sim_layout.addWidget(self.stop_btn)
        sim_layout.addWidget(self.reset_btn)
        sim_layout.addWidget(self.emergency_btn)
        
        layout.addWidget(sim_controls)
        
        # Controles de movimiento
        movement_controls = QGroupBox("🕹️ Control de Movimiento")
        movement_layout = QVBoxLayout(movement_controls)
        movement_layout.setContentsMargins(8, 15, 8, 8)
        
        # Información de teclas
        keys_info = QLabel("💡 Usa las teclas W-A-S-D o los botones (control realista de ruedas):")
        keys_info.setFont(QFont("Arial", 9))
        keys_info.setStyleSheet("color: #aaaaaa; margin: 5px;")
        keys_info.setWordWrap(True)
        movement_layout.addWidget(keys_info)
        
        # Grid de botones de movimiento
        button_grid = QGridLayout()
        button_grid.setSpacing(8)
        button_grid.setContentsMargins(20, 10, 20, 10)
        
        # Botón Adelante
        self.forward_btn = QPushButton("⬆\nAdelante\n(W)")
        self.forward_btn.pressed.connect(self.move_forward)
        self.forward_btn.released.connect(self.stop_movement)
        self.forward_btn.setMinimumSize(100, 80)
        
        # Botón Izquierda
        self.left_btn = QPushButton("⬅\nIzquierda\n(A)")
        self.left_btn.pressed.connect(self.turn_left)
        self.left_btn.released.connect(self.stop_movement)
        self.left_btn.setMinimumSize(100, 80)
        
        # Botón Derecha
        self.right_btn = QPushButton("➡\nDerecha\n(D)")
        self.right_btn.pressed.connect(self.turn_right)
        self.right_btn.released.connect(self.stop_movement)
        self.right_btn.setMinimumSize(100, 80)
        
        # Botón Atrás
        self.backward_btn = QPushButton("⬇\nAtrás\n(S)")
        self.backward_btn.pressed.connect(self.move_backward)
        self.backward_btn.released.connect(self.stop_movement)
        self.backward_btn.setMinimumSize(100, 80)
        
        # Estilo para botones de movimiento
        movement_style = """
            QPushButton {
                background-color: #607D8B;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:pressed {
                background-color: #37474F;
            }
            QPushButton:hover {
                background-color: #546E7A;
            }
        """
        
        for btn in [self.forward_btn, self.left_btn, self.right_btn, self.backward_btn]:
            btn.setStyleSheet(movement_style)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Colocar botones en grid
        button_grid.addWidget(self.forward_btn, 0, 1)
        button_grid.addWidget(self.left_btn, 1, 0)
        button_grid.addWidget(self.right_btn, 1, 2)
        button_grid.addWidget(self.backward_btn, 2, 1)
        
        movement_layout.addLayout(button_grid)
        layout.addWidget(movement_controls)
        
        # Control de velocidades
        speed_controls = QGroupBox("⚡ Control de Velocidad")
        speed_layout = QVBoxLayout(speed_controls)
        speed_layout.setContentsMargins(8, 15, 8, 8)
        
        # Velocidad lineal
        lin_speed_layout = QHBoxLayout()
        lin_label = QLabel("🏃 Velocidad Lineal:")
        lin_label.setMinimumWidth(120)
        lin_speed_layout.addWidget(lin_label)
        
        self.linear_speed_slider = QSlider(Qt.Horizontal)
        self.linear_speed_slider.setMinimum(1)
        self.linear_speed_slider.setMaximum(50)
        self.linear_speed_slider.setValue(25)
        self.linear_speed_slider.valueChanged.connect(self.change_linear_speed)
        self.linear_speed_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        self.linear_speed_label = QLabel("2.5 m/s")
        self.linear_speed_label.setMinimumWidth(60)
        self.linear_speed_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        
        lin_speed_layout.addWidget(self.linear_speed_slider)
        lin_speed_layout.addWidget(self.linear_speed_label)
        speed_layout.addLayout(lin_speed_layout)
        
        # Velocidad angular
        ang_speed_layout = QHBoxLayout()
        ang_label = QLabel("🔄 Velocidad Angular:")
        ang_label.setMinimumWidth(120)
        ang_speed_layout.addWidget(ang_label)
        
        self.angular_speed_slider = QSlider(Qt.Horizontal)
        self.angular_speed_slider.setMinimum(1)
        self.angular_speed_slider.setMaximum(40)
        self.angular_speed_slider.setValue(25)
        self.angular_speed_slider.valueChanged.connect(self.change_angular_speed)
        self.angular_speed_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        self.angular_speed_label = QLabel("2.5 rad/s")
        self.angular_speed_label.setMinimumWidth(60)
        self.angular_speed_label.setStyleSheet("font-weight: bold; color: #FF9800;")
        
        ang_speed_layout.addWidget(self.angular_speed_slider)
        ang_speed_layout.addWidget(self.angular_speed_label)
        speed_layout.addLayout(ang_speed_layout)
        
        layout.addWidget(speed_controls)
        
        # Control de luces
        lights_controls = QGroupBox("💡 Control de Luces")
        lights_layout = QVBoxLayout(lights_controls)
        lights_layout.setContentsMargins(8, 15, 8, 8)
        
        # Luces individuales
        self.light_checkboxes = {}
        light_names = {
            'front': '🔦 Luces Frontales',
            'back': '🔴 Luces Traseras', 
            'left': '🟠 Luz Izquierda',
            'right': '🟠 Luz Derecha'
        }
        
        lights_grid = QGridLayout()
        row = 0
        for light_key, light_label in light_names.items():
            checkbox = QCheckBox(light_label)
            checkbox.stateChanged.connect(
                lambda state, key=light_key: self.toggle_light(key, state == 2)
            )
            checkbox.setStyleSheet("QCheckBox { font-size: 10px; }")
            self.light_checkboxes[light_key] = checkbox
            lights_grid.addWidget(checkbox, row // 2, row % 2)
            row += 1
        
        lights_layout.addLayout(lights_grid)
        
        # Separador
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #3a3a45;")
        lights_layout.addWidget(line)
        
        # Luces de emergencia
        self.warning_btn = QPushButton("🚨 Luces de Emergencia")
        self.warning_btn.setCheckable(True)
        self.warning_btn.clicked.connect(self.toggle_warning_lights)
        self.warning_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF5722;
                color: white;
                font-weight: bold;
                border-radius: 6px;
                padding: 8px;
            }
            QPushButton:checked {
                background-color: #D32F2F;
            }
            QPushButton:hover {
                background-color: #E64A19;
            }
            QPushButton:pressed {
                background-color: #C41E1E;
            }
        """)
        lights_layout.addWidget(self.warning_btn)
        
        layout.addWidget(lights_controls)
        
        # Espaciador final
        layout.addStretch()
        
        return panel

    def create_info_panel(self):
        panel = QGroupBox("📊 Estado del Sistema y Telemetría")
        panel.setMinimumWidth(400)
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # Estado de la simulación
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("🔧 Estado:"))
        self.status_label = QLabel("Detenido")
        self.status_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.status_label.setStyleSheet("""
            QLabel {
                color: #ff6b6b;
                background-color: #3a2d2d;
                border-radius: 4px;
                padding: 4px 8px;
            }
        """)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)
        
        # Información del robot actualizada
        robot_info = QLabel("""
        <h3 style="color: #bbbbff;">🚛 Robot de Carga Rectangular - Control Realista de Ruedas</h3>
        <b style="color: #aaccff;">🔧 Especificaciones Técnicas:</b><br>
        • Dimensiones: 1.0 x 0.6 x 0.24 m<br>
        • Peso total: 6.8 kg (5kg cuerpo + 1.8kg ruedas)<br>
        • 6 ruedas independientes Ø24cm (radio 0.12m), 2 motorizadas en medio, 4 pasivas<br>
        • Control diferencial por 2 ruedas motorizadas<br>
        • Ejes de rotación corregidos (eje Z)<br><br>
        
        <b style="color: #aaccff;">⚙️ Sistema de Control:</b><br>
        • Controlador PID por rueda individual (Kp=80, Ki=0.5, Kd=2.0)<br>
        • Cinemática diferencial realista<br>
        • Límites de aceleración (20 rad/s²)<br>
        • Fricción y física simulada con PyBullet<br><br>
        
        <b style="color: #aaccff;">🎮 Controles:</b><br>
        • <b>W</b>: Avanzar | <b>S</b>: Retroceder<br>
        • <b>A</b>: Girar izquierda | <b>D</b>: Girar derecha<br>
        • 🛑 Botón de emergencia para parada inmediata
        """)
        robot_info.setWordWrap(True)
        robot_info.setStyleSheet("""
            QLabel {
                background-color: #25252d;
                border: 1px solid #3a3a45;
                border-radius: 6px;
                padding: 10px;
                font-size: 10px;
            }
        """)
        layout.addWidget(robot_info)
        
        # Telemetría en tiempo real mejorada
        telemetry_group = QGroupBox("📡 Telemetría en Tiempo Real")
        telemetry_layout = QVBoxLayout(telemetry_group)
        telemetry_layout.setContentsMargins(8, 15, 8, 8)
        
        self.telemetry_label = QLabel("🔄 Esperando datos de telemetría...")
        self.telemetry_label.setFont(QFont("Consolas", 9))
        self.telemetry_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a22;
                color: #00ff00;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Courier New', monospace;
            }
        """)
        self.telemetry_label.setWordWrap(True)
        self.telemetry_label.setMinimumHeight(150)
        telemetry_layout.addWidget(self.telemetry_label)
        
        layout.addWidget(telemetry_group)
        
        # Estado de las ruedas con barras de progreso mejoradas
        wheels_group = QGroupBox("🔧 Estado de las Ruedas Motorizadas")
        wheels_layout = QVBoxLayout(wheels_group)
        wheels_layout.setContentsMargins(8, 15, 8, 8)
        
        self.wheel_bars = []
        self.wheel_speed_labels = []
        wheel_names = ["Izquierda Motorizada", "Derecha Motorizada"]
        
        for i, name in enumerate(wheel_names):
            wheel_container = QHBoxLayout()
            
            # Etiqueta de la rueda
            wheel_label = QLabel(f"{name}:")
            wheel_label.setMinimumWidth(150)
            wheel_label.setFont(QFont("Arial", 9, QFont.Bold))
            wheel_container.addWidget(wheel_label)
            
            # Barra de progreso
            progress_bar = QProgressBar()
            progress_bar.setMinimum(-100)
            progress_bar.setMaximum(100)
            progress_bar.setValue(0)
            progress_bar.setTextVisible(True)
            progress_bar.setFormat("%v%")
            progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid #3a3a45;
                    border-radius: 5px;
                    text-align: center;
                    background-color: #25252d;
                }
                QProgressBar::chunk {
                    border-radius: 3px;
                }
            """)
            
            self.wheel_bars.append(progress_bar)
            wheel_container.addWidget(progress_bar)
            
            # Etiqueta de velocidad
            speed_label = QLabel("0.0 rad/s")
            speed_label.setMinimumWidth(70)
            speed_label.setFont(QFont("Arial", 8))
            speed_label.setStyleSheet("color: #aaaaaa;")
            self.wheel_speed_labels.append(speed_label)
            wheel_container.addWidget(speed_label)
            
            wheels_layout.addLayout(wheel_container)
        
        layout.addWidget(wheels_group)
        
        # Estado de las luces
        lights_status_group = QGroupBox("💡 Estado de Luces")
        lights_status_layout = QVBoxLayout(lights_status_group)
        lights_status_layout.setContentsMargins(8, 15, 8, 8)
        
        self.lights_status = QLabel()
        self.lights_status.setStyleSheet("""
            QLabel {
                background-color: #25252d;
                border: 1px solid #3a3a45;
                border-radius: 4px;
                padding: 6px;
                font-size: 9px;
            }
        """)
        self.update_lights_display()
        lights_status_layout.addWidget(self.lights_status)
        
        layout.addWidget(lights_status_group)
        
        # Espaciador final
        layout.addStretch()
        
        return panel

    def setup_keyboard_shortcuts(self):
        """Configurar atajos de teclado para W-A-S-D"""
        self.shortcuts = {}
        
        # Crear atajos para W, A, S, D
        for key in ['W', 'A', 'S', 'D']:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda k=key.lower(): self.key_pressed(k))
            self.shortcuts[key] = shortcut
        
        print("⌨️ Atajos de teclado configurados: W-A-S-D")

    def key_pressed(self, key):
        """Manejar pulsaciones de tecla"""
        if not self.simulator.simulation_running:
            return
            
        if key == 'w':
            self.simulator.move_forward()
        elif key == 's':
            self.simulator.move_backward()
        elif key == 'a':
            self.simulator.turn_left()
        elif key == 'd':
            self.simulator.turn_right()

    def keyReleaseEvent(self, event):
        """Manejar liberación de teclas"""
        key = event.text().lower()
        if key in ['w', 'a', 's', 'd'] and self.simulator.simulation_running:
            self.simulator.stop_movement()

    # Métodos de control de movimiento
    def move_forward(self):
        self.simulator.move_forward()

    def move_backward(self):
        self.simulator.move_backward()

    def turn_left(self):
        self.simulator.turn_left()

    def turn_right(self):
        self.simulator.turn_right()

    def stop_movement(self):
        self.simulator.stop_movement()

    def emergency_stop(self):
        self.simulator.emergency_stop()
        print("🚨 PARADA DE EMERGENCIA ACTIVADA")

    # Métodos de control de luces
    def toggle_light(self, light_name, state):
        self.simulator.set_light_state(light_name, state)
        self.update_lights_display()

    def toggle_warning_lights(self):
        self.simulator.toggle_warning_lights()
        # Actualizar checkboxes
        warning_state = self.simulator.lights_state['warning']
        self.light_checkboxes['left'].setChecked(warning_state)
        self.light_checkboxes['right'].setChecked(warning_state)
        self.update_lights_display()

    # Métodos de control de velocidad
    def change_linear_speed(self, value):
        speed = value / 10.0  # Convertir a m/s
        angular_speed = self.simulator.max_angular_speed
        self.simulator.set_max_speeds(speed, angular_speed)
        self.linear_speed_label.setText(f"{speed:.1f} m/s")

    def change_angular_speed(self, value):
        angular_speed = value / 10.0  # Convertir a rad/s
        linear_speed = self.simulator.max_linear_speed
        self.simulator.set_max_speeds(linear_speed, angular_speed)
        self.angular_speed_label.setText(f"{angular_speed:.1f} rad/s")

    def update_lights_display(self):
        """Actualizar el display de estado de luces"""
        lights_text = "<b style='color: #bbbbff;'>Estado Actual:</b><br>"
        for light, state in self.simulator.lights_state.items():
            if light == 'warning':
                continue
            status = "🟢 ENCENDIDA" if state else "🔴 APAGADA"
            lights_text += f"• <b>{light.title()}</b>: {status}<br>"
        
        if self.simulator.lights_state['warning']:
            lights_text += "<br><b style='color: #ff6b6b;'>🚨 LUCES DE EMERGENCIA ACTIVAS</b>"
        
        self.lights_status.setText(lights_text)

    def update_telemetry(self):
        """Actualiza la telemetría del robot con más detalles"""
        if self.simulator.simulation_running:
            robot_state = self.simulator.get_robot_state()
            
            # Información de posición y velocidad
            pos = robot_state['position']
            vel = robot_state['linear_velocity']
            ang_vel = robot_state['angular_velocity']
            euler_orn = robot_state['euler_orientation']
            
            # Velocidades de ruedas
            wheel_velocities = [wheel_state['velocity'] for wheel_state in robot_state['wheels']]
            
            # Formatear telemetría mejorada
            telemetry_text = f"""📍 POSICIÓN: X={pos[0]:+6.2f}m  Y={pos[1]:+6.2f}m  Z={pos[2]:+6.2f}m
🔄 ORIENTACIÓN: Yaw={euler_orn[2]:+6.2f} rad
🏃 VELOCIDAD LIN: {np.linalg.norm(vel):6.2f} m/s
🌀 VELOCIDAD ANG: {np.linalg.norm(ang_vel):6.2f} rad/s
🎯 COMANDOS: Lin={robot_state['target_linear_vel']:+5.2f} m/s  Ang={robot_state['target_angular_vel']:+5.2f} rad/s

🔧 RUEDAS MOTORIZADAS (rad/s):
   L: {wheel_velocities[0]:+6.2f}  |  R: {wheel_velocities[1]:+6.2f}"""
            
            self.telemetry_label.setText(telemetry_text)
            
            # Actualizar barras de ruedas
            max_wheel_vel = self.simulator.controller.max_wheel_velocity
            for i, vel in enumerate(wheel_velocities):
                percentage = int((vel / max_wheel_vel) * 100)
                self.wheel_bars[i].setValue(percentage)
                
                # Actualizar etiqueta de velocidad
                self.wheel_speed_labels[i].setText(f"{vel:+5.1f} rad/s")
                
                # Cambiar color según la velocidad
                if abs(percentage) > 70:
                    color = "#f44336"  # Rojo
                elif abs(percentage) > 40:
                    color = "#FF9800"  # Naranja
                else:
                    color = "#4CAF50"  # Verde
                
                self.wheel_bars[i].setStyleSheet(f"""
                    QProgressBar::chunk {{
                        background-color: {color};
                        border-radius: 3px;
                    }}
                """)
        else:
            self.telemetry_label.setText("⏸️ Simulación detenida - No hay datos de telemetría")

    # Métodos de control de simulación
    def start_simulation(self):
        self.simulator.start_simulation()
        self.status_label.setText("Ejecutándose")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #4CAF50;
                background-color: #2d3a2d;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
            }
        """)
        print("▶️ Simulación iniciada desde interfaz")

    def stop_simulation(self):
        self.simulator.stop_simulation()
        self.status_label.setText("Detenido")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #ff6b6b;
                background-color: #3a2d2d;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
            }
        """)
        print("⏸️ Simulación detenida desde interfaz")

    def reset_robot(self):
        self.simulator.reset_robot()
        # Reset checkboxes
        for checkbox in self.light_checkboxes.values():
            checkbox.setChecked(False)
        self.warning_btn.setChecked(False)
        print("🔄 Robot reiniciado desde interfaz")

    def update_simulation(self):
        """Bucle principal de actualización"""
        self.simulator.step_simulation()
        
        # Actualizar telemetría cada ciertos frames para mejor rendimiento
        if hasattr(self, '_telemetry_counter'):
            self._telemetry_counter += 1
        else:
            self._telemetry_counter = 0
            
        if self._telemetry_counter % 10 == 0:  # Cada ~166ms para más fluidez
            self.update_telemetry()
            
        if self._telemetry_counter % 60 == 0:  # Cada 1 segundo
            self.update_lights_display()

    def closeEvent(self, event):
        """Manejar cierre de ventana"""
        print("👋 Cerrando aplicación...")
        self.simulator.disconnect()
        event.accept()

    def resizeEvent(self, event):
        """Manejar redimensionamiento para responsive"""
        super().resizeEvent(event)
        # Ajustar elementos según el tamaño de la ventana
        width = self.width()
        
        # Solo ajustar si ya tenemos los widgets inicializados
        if hasattr(self, 'main_layout') and self.main_layout:
            # Ajustar diseño para pantallas pequeñas
            if width < 1000:
                # Reducir márgenes y espaciado
                self.main_layout.setSpacing(8)
                for panel in self.findChildren(QGroupBox):
                    panel.layout().setContentsMargins(6, 12, 6, 6)
            else:
                # Restaurar márgenes normales
                self.main_layout.setSpacing(15)
                for panel in self.findChildren(QGroupBox):
                    panel.layout().setContentsMargins(8, 15, 8, 8)

def main():
    """Función principal"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Usar estilo moderno
    
    # Configurar información de la aplicación
    app.setApplicationName("Robot de Carga Simulado")
    app.setApplicationVersion("2.2")  # Actualizado
    app.setOrganizationName("Simulación PyBullet")
    
    try:
        # Crear e mostrar la ventana principal
        window = CargoRobotInterface()
        window.show()
        
        print("🎮 Interfaz de control iniciada")
        print("💡 Usa W-A-S-D para controlar el robot")
        print("🚨 Botón de emergencia disponible para parada inmediata")
        print("🔧 Ruedas corregidas - orientación eje Z, 6 ruedas con 2 motorizadas")
        
        # Ejecutar aplicación
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"❌ Error al inicializar la aplicación: {e}")
        print("📦 Asegúrate de tener instaladas las dependencias:")
        print("   pip install pybullet pyside6 numpy")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()