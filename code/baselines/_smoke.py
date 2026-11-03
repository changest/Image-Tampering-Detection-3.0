import os
import cv2
import torch
import numpy as np
from PIL import Image

from baselines.eval_baselines import load_cnndetection, load_univfd, _forward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
real_dir = r"E:\datasets\tigas\val\real"
fake_dir = r"E:\datasets\tigas\val\fake"
real_file = os.path.join(real_dir, sorted(os.listdir(real_dir))[0])
fake_file = os.path.join(fake_dir, sorted(os.listdir(fake_dir))[0])


def load_rgb(p):
    return Image.fromarray(cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))


for name, loader in (("cnndetection", load_cnndetection), ("univfd", load_univfd)):
    model, tf = loader(device)
    for tag, p in (("real", real_file), ("fake", fake_file)):
        x = tf(load_rgb(p)).unsqueeze(0)
        logit = _forward(name, model, x, device)
        print(f"{name} {tag}: logit={float(logit):.3f} prob={float(torch.sigmoid(logit)):.4f}", flush=True)
print("smoke OK")
