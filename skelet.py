import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Загружаем модель MoveNet Lightning (скоростная версия для single-pose)
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures["serving_default"]

def detect_movenet(frame: np.ndarray):
    """
    Принимает кадр BGR (numpy), возвращает массив shape (17, 3):
        [[y_rel, x_rel, score], …] 17 точек в относительных координатах.
    """
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # Приводим к размеру 192×192 с паддингом
    inp = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 192)
    inp = tf.cast(inp, dtype=tf.int32)

    # Инференс
    outputs = movenet(inp)
    # Получаем тензор shape (1, 1, 17, 3)
    keypoints_with_scores = outputs["output_0"].numpy()
    keypoints = keypoints_with_scores[0, 0, :, :]  # (17,3)
    return keypoints  # y, x, score

def draw_movenet(frame: np.ndarray, keypoints: np.ndarray, threshold=0.3):
    """
    Рисует скелет на кадре по ключевым точкам MoveNet.
    keypoints: np.array shape (17, 3) – [y_rel, x_rel, score] у каждой точки.
    """
    H, W, _ = frame.shape

    # Список соединений для COCO-формата (17 точек)
    edges = [
        (0,1), (1,3), (0,2), (2,4),
        (0,5), (5,7), (7,9),
        (0,6), (6,8), (8,10),
        (5,6), (5,11), (6,12),
        (11,12), (11,13), (13,15),
        (12,14), (14,16)
    ]

    # Преобразуем относительные координаты (0..1) в пиксели
    pts = []
    for i in range(17):
        y, x, score = keypoints[i]
        if score < threshold:
            pts.append(None)
        else:
            px = int(x * W)
            py = int(y * H)
            pts.append((px, py))

    # Рисуем линии (косточки)
    for (u, v) in edges:
        if pts[u] is not None and pts[v] is not None:
            cv2.line(frame, pts[u], pts[v], (0, 255, 0), 2)

    # Рисуем кружочки в каждой точке
    for p in pts:
        if p is not None:
            cv2.circle(frame, p, 4, (0, 0, 255), -1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Получаем 17 ключевых точек MoveNet
        keypoints = detect_movenet(frame)

        # 2) Наносим скелет на кадр
        draw_movenet(frame, keypoints, threshold=0.3)

        # 3) Здесь можно встроить ваш детектор товара/человека, трекинг и AR

        # 4) Отображаем результат
        cv2.imshow("MoveNet Pose", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()