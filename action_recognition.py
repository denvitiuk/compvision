import torch
import numpy as np
import cv2
import torch.nn.functional as F
from pytorchvideo.models.hub import slowfast_r50

# Инициализация модели SlowFast (предобученной на Kinetics400)
device = "cuda" if torch.cuda.is_available() else "cpu"
action_model = slowfast_r50(pretrained=True).eval().to(device)

# Загрузка меток Kinetics-400 (по одной на строку)
with open("kinetics_labels.txt", "r") as f:
    kinetics_labels = [line.strip() for line in f if line.strip()]

def recognize_clip(frames_list):
    """
    Принимает список NumPy-кадров длиной T (теперь T=32),
    возвращает топ-5 действий [(label, prob), ...].
    """
    # 1. BGR → RGB и resize до 256×256
    clip_rgb = []
    for frame in frames_list:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        clip_rgb.append(img)

    # 2. (T, H, W, 3) → (1, 3, T, 256, 256)
    clip_np = np.stack(clip_rgb, axis=0)                                  # (32, 256, 256, 3)
    clip_tensor = torch.from_numpy(clip_np).permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, 32, 256, 256)
    clip_tensor = clip_tensor.to(device).float() / 255.0

    # 3. Формируем два “потока” для SlowFast
    fast_pathway = clip_tensor                                      # (1, 3, 32, 256, 256)
    slow_pathway = clip_tensor[:, :, ::4, :, :]                     # (1, 3, 8, 256, 256)

    inputs = [slow_pathway, fast_pathway]                           # список из двух тензоров

    # 4. Инференс
    with torch.no_grad():
        preds = action_model(inputs)                                # (1, 400) logits
        probs = F.softmax(preds, dim=1)[0]                          # (400,) вероятности

    # 5. Топ-5
    top5_prob, top5_id = torch.topk(probs, k=5)
    return [(kinetics_labels[int(i)], float(p)) for i, p in zip(top5_id, top5_prob)]