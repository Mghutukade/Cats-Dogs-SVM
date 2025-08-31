import os
import cv2
import numpy as np

def load_data(data_dir, img_size=64):
    X, y = [], []
    for label, class_name in enumerate(["cat", "dog"]):
        folder = os.path.join(data_dir, class_name)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (img_size, img_size))
                X.append(img.flatten())  # flatten into 1D vector
                y.append(label)
            except Exception as e:
                print("Error loading image:", e)
    return np.array(X), np.array(y)
