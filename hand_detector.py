import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "hand_landmarker.task"

# Inicializa o MediaPipe
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)

options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
)

detector = mp_vision.HandLandmarker.create_from_options(options)


def detectar_mao(frame):
    """
    Recebe um frame da webcam e retorna:

    landmarks -> vetor de 63 posições (ou None)
    hand       -> landmarks do MediaPipe para desenhar
    """

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    resultado = detector.detect(mp_image)

    if not resultado.hand_landmarks:
        return None, None

    hand = resultado.hand_landmarks[0]

    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand],
        dtype=np.float32
    )

    # Centraliza no pulso
    wrist = coords[0].copy()
    coords -= wrist

    # Normaliza pela maior distância
    escala = np.max(np.linalg.norm(coords, axis=1))

    if escala > 1e-8:
        coords /= escala

    vetor = coords.flatten()

    return vetor, hand