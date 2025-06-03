import cv2
import torch
import torchvision
import torchvision.transforms as T
import time

def main():
    # Определяем устройство: сначала MPS (для Apple Silicon), затем CUDA, иначе CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Загружаем предобученную модель SSD300 VGG16 (легкая альтернатива)
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    model.to(device)
    model.eval()

    # Список классов COCO
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
        'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]

    transform = T.Compose([T.ToTensor()])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть камеру!")
        return

    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)

    prev_time = time.time()
    frame_count = 0
    process_every = 3  # обрабатываем каждый 3-й кадр
    detection_results = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка захвата кадра!")
            break

        # Уменьшаем размер кадра для ускорения обработки
        resized_frame = cv2.resize(frame, (640, 480))
        frame_count += 1

        if frame_count % process_every == 0:
            # Преобразуем изображение из BGR в RGB и в тензор
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img_tensor = transform(rgb_frame).unsqueeze(0).to(device)
            start_time = time.time()
            with torch.no_grad():
                predictions = model(img_tensor)[0]
            inference_time = time.time() - start_time
            print(f"Inference time: {inference_time:.2f} sec")
            # Сохраняем результаты детекции
            detection_results = (predictions['boxes'].cpu().numpy(),
                                 predictions['labels'].cpu().numpy(),
                                 predictions['scores'].cpu().numpy())
        else:
            if detection_results is not None:
                boxes, labels, scores = detection_results
            else:
                boxes, labels, scores = ([], [], [])

        # Отрисовка детекций
        conf_threshold = 0.7
        for i, score in enumerate(scores):
            if score >= conf_threshold:
                box = boxes[i]
                cls_id = labels[i]
                cls_name = COCO_INSTANCE_CATEGORY_NAMES[cls_id]
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(resized_frame, f"{cls_name} {score:.2f}",
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Расчет и вывод общего FPS (включая задержку между кадрами)
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(resized_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Detections", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
