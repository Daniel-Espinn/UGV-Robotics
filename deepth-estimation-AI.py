import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import time

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Cargar modelo MiDaS
model_type = "MiDaS_small"  # Usamos un modelo ligero para mejor rendimiento en CPU
midas = torch.hub.load("intel-isl/MiDaS", model_type, force_reload=True)
midas.to(device).eval()

# Definir transformaciones manualmente para evitar problemas con midas_transforms
def midas_transform(image):
    # Convertir PIL Image a tensor y normalizar
    transform = transforms.Compose([
        transforms.Resize((384, 384)),  # Tamaño divisible por 32
        transforms.ToTensor(),  # Convertir a tensor [C, H, W]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalización
    ])
    return transform(image)

# Inicializar captura de cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

try:
    while True:
        # Leer frame de la cámara
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame.")
            break

        # Obtener dimensiones originales
        img_height, img_width = frame.shape[:2]
        start = time.time()

        # Convertir de BGR a RGB y luego a PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Aplicar transformaciones
        input_tensor = midas_transform(frame_pil).to(device)

        # Realizar inferencia
        with torch.no_grad():
            prediction = midas(input_tensor.unsqueeze(0))  # [1, H, W]
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),  # [1, 1, H, W]
                size=(img_height, img_width),  # Redimensionar a resolución original
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

        # Normalizar el mapa de profundidad para visualización
        depth_map = cv2.normalize(prediction, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

        # Calcular y mostrar FPS
        fps = 1 / (time.time() - start)
        cv2.putText(frame, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar imágenes
        cv2.imshow("Imagen Original", frame)
        cv2.imshow("Mapa de Profundidad", depth_map_colored)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print(f"Error durante la ejecución: {e}")

finally:
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()