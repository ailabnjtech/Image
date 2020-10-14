
import numpy as np

from keras.utils import to_categorical

import os
import cv2
from keras.applications.vgg16 import preprocess_input
prepath="./tea/"
classes = os.listdir(prepath)  # 类别序号和名称
datas = []
labels = []
for i, abspath in enumerate(classes):  # prepath的每一个文件目录
    img_names = os.listdir(prepath + abspath)
    for img_name in img_names:  # 子目录中的每一张图片
        img = cv2.imread(os.path.join(prepath + abspath, img_name))  # cv2读取
        if not isinstance(img, np.ndarray):
            print("read img error")
            continue
        img = cv2.resize(img, (224, 224))  # 尺寸变换224*224
        # img = img.astype(np.float32)  # 类型转换为float32
        img = preprocess_input(img)
        label = to_categorical(i, 5)
        labels.append(label)
        datas.append(img)
datas = np.array(datas)
labels = np.array(labels)
np.save("datas/train_data.npy", datas)
np.save("label/labels.npy", labels)