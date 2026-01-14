import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split



input_dir = "c:/Users/MONSTER/Masaüstü/computer vision/beginner/clf-data/"
categories = ["empty", "not_empty"]

data = []
labels = []
for category_idx,category in enumerate(categories):
    for file in os.listdir(input_dir + category):
        img_path = os.path.join(input_dir,category, file)
        img = imread(img_path)
        img =resize(img, (15,15))
        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)