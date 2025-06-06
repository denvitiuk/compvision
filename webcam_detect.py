import time
import cv2
import numpy as np
import random
from collections import deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from action_recognition import recognize_clip  # модуль для action recognition

# ПУТИ К ВАШИМ МОДЕЛЯМ
PRODUCT_MODEL_PATH = "/Users/yuliyanatasheva/PycharmProjects/compvision/runs/detect/train22/weights/best.pt"
PERSON_MODEL_PATH  = "yolov8n.pt"

# ЗАГРУЗКА МОДЕЛЕЙ
product_model = YOLO(PRODUCT_MODEL_PATH)
person_model  = YOLO(PERSON_MODEL_PATH)

# ЗАГРУЗКА ИМЁН КЛАССОВ ДЛЯ ТОВАРОВ
with open("classes.txt", "r") as f:
    product_classes = [line.strip() for line in f if line.strip()]

# ИНИЦИАЛИЗАЦИЯ ТРЕКЕРА (Deep SORT)
deepsort = DeepSort(max_age=30, n_init=2, nms_max_overlap=0.5)

# Словарь для хранения треков, времени и буфера кадров (буфер = 32 кадра)
# Формат: { track_id: {"label": str, "start_time": float, "last_seen": float, "buffer": deque([...])} }
active_tracks = {}

# Словарь для случайных цветов для каждого трека
track_colors = {}

# Порог уверенности для распознавания действия
ACTION_THRESHOLD = 0.7

# Инициализация видеопотока (0 - веб-камера)
cap = cv2.VideoCapture(0)
frame_id = 0

while True:
    loop_start = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    current_time = time.time()

    #### 1) ДЕТЕКЦИЯ ТОВАРОВ ####
    product_boxes, product_confs, product_labels, product_cls_ids = [], [], [], []
    product_results = product_model.predict(frame, conf=0.5, verbose=False)
    if product_results and product_results[0].boxes is not None:
        for box in product_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = "unknown_product"
            if 0 <= cls_id < len(product_classes):
                class_name = product_classes[cls_id]
            w, h = x2 - x1, y2 - y1
            product_boxes.append([x1, y1, w, h])
            product_confs.append(conf)
            product_labels.append(class_name)
            product_cls_ids.append(cls_id)

    #### 2) ДЕТЕКЦИЯ ЛЮДЕЙ ####
    person_boxes, person_confs, person_labels, person_cls_ids = [], [], [], []
    person_results = person_model.predict(frame, conf=0.5, verbose=False)
    if person_results and person_results[0].boxes is not None:
        for box in person_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            if person_model.names[cls_id].lower() != "person":
                continue
            w, h = x2 - x1, y2 - y1
            person_boxes.append([x1, y1, w, h])
            person_confs.append(conf)
            person_labels.append("person")
            person_cls_ids.append(-1)

    #### 3) ОБЪЕДИНЕНИЕ ДЕТЕКЦИЙ ####
    all_boxes = product_boxes + person_boxes
    all_confs = product_confs + person_confs
    all_labels = product_labels + person_labels
    all_cls_ids = product_cls_ids + person_cls_ids

    detections = []
    for i in range(len(all_boxes)):
        detections.append([all_boxes[i], all_confs[i], all_cls_ids[i]])

    #### 4) ОБНОВЛЕНИЕ ТРЕКЕРА ####
    tracks = deepsort.update_tracks(detections, frame=frame)

    #### 5) ОТРИСОВКА И ACTION RECOGNITION ####
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Определяем метку через IoU
        best_iou, best_label = 0, None
        for i, box in enumerate(all_boxes):
            bx, by, bw, bh = box
            bx2, by2 = bx + bw, by + bh
            ix1, iy1 = max(x1, int(bx)), max(y1, int(by))
            ix2, iy2 = min(x2, int(bx2)), min(y2, int(by2))
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            if iw * ih == 0:
                continue
            area_track = (x2 - x1) * (y2 - y1)
            area_box = bw * bh
            union = area_track + area_box - (iw * ih)
            iou = (iw * ih) / union if union > 0 else 0
            if iou > best_iou:
                best_iou, best_label = iou, all_labels[i]
        if not best_label:
            best_label = "unknown"

        # Если трек новый, создаём запись с буфером на 32 кадра
        if track_id not in active_tracks:
            active_tracks[track_id] = {
                "label": best_label,
                "start_time": current_time,
                "last_seen": current_time,
                "buffer": deque(maxlen=32)
            }
        else:
            active_tracks[track_id]["last_seen"] = current_time
            active_tracks[track_id]["label"] = best_label

        time_in_scene = current_time - active_tracks[track_id]["start_time"]
        display_text = f"{best_label} (ID={track_id}) {time_in_scene:.1f}s"

        # Action Recognition только для людей
        if best_label == "person":
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size != 0:
                buf = active_tracks[track_id]["buffer"]
                resized = cv2.resize(person_crop, (256, 256))
                buf.append(resized)
                print(f"Track {track_id}: буфер заполнен {len(buf)}/32 кадров")

                # Когда накопилось 32 кадра, запускаем распознавание
                if len(buf) == 32:
                    print(f"Track {track_id}: запускаем recognize_clip()")
                    top5 = recognize_clip(list(buf))
                    print(f"Track {track_id}: топ-5 меток ↔ {top5}")
                    top1_name, top1_prob = top5[0]
                    print(f"Track {track_id}: результат action recognition → {top1_name} ({top1_prob:.2f})")

                    # Запись в файл <class_name>.txt
                    filename = f"{top1_name}.txt"
                    with open(filename, "a") as log_f:
                        log_f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                                    f"Track={track_id} Prob={top1_prob:.2f}\n")

                    # Если top1 — “взятие/держание”
                    if top1_name.lower() in ["picking", "holding_object", "hold"] and top1_prob > ACTION_THRESHOLD:
                        print(f"[Action] Track {track_id} — {top1_name} ({top1_prob:.2f})")
                    else:
                        print(f"[Action] Track {track_id}: ничего не распознано (или prob < {ACTION_THRESHOLD})")

                    buf.clear()

        # Назначаем цвет и рисуем bounding-box + текст
        if track_id not in track_colors:
            track_colors[track_id] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        color = track_colors[track_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, display_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Удаляем “пропавшие” треки
    current_track_ids = {t.track_id for t in tracks if t.is_confirmed()}
    to_remove = []
    for tid, info in active_tracks.items():
        if tid not in current_track_ids and (current_time - info["last_seen"]) > 3.0:
            print(f"Track {tid} ({info['label']}) duration: {info['last_seen'] - info['start_time']:.1f}s")
            to_remove.append(tid)
    for tid in to_remove:
        del active_tracks[tid]
        if tid in track_colors:
            del track_colors[tid]

    #### 6) ВЫЧИСЛЕНИЕ FPS ####
    fps = 1.0 / (time.time() - loop_start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Combined Tracking + Action Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Выход по ESC
        break

cap.release()
cv2.destroyAllWindows()
