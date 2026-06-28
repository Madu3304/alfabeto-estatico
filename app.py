import cv2
import joblib
import numpy as np

from collections import Counter, deque
from hand_detector import detectar_mao

# Conexões entre os landmarks da mão

CONEXOES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),          # Indicador
    (5, 9), (9, 10), (10, 11), (11, 12),     # Médio
    (9, 13), (13, 14), (14, 15), (15, 16),   # Anelar
    (13, 17), (17, 18), (18, 19), (19, 20),  # Mindinho
    (0, 17)                                  # Palma
]

# Configurações

HISTORICO_MAX = 10
MIN_VOTOS = 7

historico = deque(maxlen=HISTORICO_MAX)

# Carrega modelo

modelo = joblib.load("modelo/classificador.pkl")

label_map = np.load(
    "modelo/label_map.npy",
    allow_pickle=True
).item()

idx_to_label = {v: k for k, v in label_map.items()}

# Webcam

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

# Loop principal

while True:

    ok, frame = camera.read()

    if not ok:
        break

    # Corrige o espelhamento
    frame = cv2.flip(frame, 1)

    # Detecta a mão
    vetor, hand = detectar_mao(frame)

    if hand is not None:

        altura, largura = frame.shape[:2]

        pontos = []

        for lm in hand:

            x = int(lm.x * largura)
            y = int(lm.y * altura)

            pontos.append((x, y))

        for inicio, fim in CONEXOES:

            cv2.line(
                frame,
                pontos[inicio],
                pontos[fim],
                (0, 255, 0),
                2
            )

        for x, y in pontos:

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

        # Classificação

        entrada = vetor.reshape(1, -1)

        predicao = modelo.predict(entrada)[0]

        historico.append(predicao)

        contador = Counter(historico)

        mais_comum, quantidade = contador.most_common(1)[0]

        # Exibição

        if quantidade >= MIN_VOTOS:

            letra = idx_to_label[mais_comum]

            cv2.putText(
                frame,
                "LETRA",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                letra,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.8,
                (0, 255, 0),
                5,
            )

        else:

            cv2.putText(
                frame,
                "Reconhecendo...",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

    else:

        historico.clear()

        cv2.putText(
            frame,
            "Nenhuma mao detectada",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Reconhecimento de Libras", frame)

    tecla = cv2.waitKey(1)

    if tecla == 27:  # ESC
        break

camera.release()
cv2.destroyAllWindows()