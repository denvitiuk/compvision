from ultralytics import YOLO


def main():
    # Загружаем базовую детекционную модель
    model = YOLO("yolov8s.pt")

    # Запускаем обучение с указанным файлом data.yaml
    results = model.train(
        data="/Users/yuliyanatasheva/PycharmProjects/compvision/datasets/data.yaml",  # путь к data.yaml
        epochs=50,  # увеличенное количество эпох для лучшего обучения
        imgsz=640,
        batch=8,
        name="train2"  # имя эксперимента, будет создана папка runs/detect/train2
    )

    # После обучения можно запустить валидацию и вывести метрики
    metrics = model.val()
    print("Результаты валидации:", metrics)


if __name__ == "__main__":
    main()
