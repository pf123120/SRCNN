import os
import cv2
import h5py
import numpy as np

train_path = "D:/python/srcnn/dataset/Train"

rand_num = 100

def preprocess_data(path):
    names = os.listdir(path)
    names = sorted(names)
    nums = names.__len__()

    data = np.zeros((nums * rand_num, 32, 32, 1), dtype=np.double)
    label = np.zeros((nums * rand_num, 20, 20, 1), dtype=np.double)

    for i in range(nums):
        img_path = os.path.join(path, names[i])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)    # 打开图片
        height, width, _ = img.shape   # 获取图片的大小
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)    # BGR转YCrCb
        y, _, _ = cv2.split(img)    # 分离出Y通道

        lr_img = cv2.resize(y, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
        lr_img = cv2.resize(lr_img, (width, height), interpolation=cv2.INTER_CUBIC)

        for j in range(rand_num):
            m = np.random.randint(0, min(width, height) - 32)
            n = np.random.randint(0, min(width, height) - 32)

            lr_patch = lr_img[m:m + 32, n:n + 32]
            hr_patch = y[m:m + 32, n:n + 32]

            lr_patch = lr_patch.astype(float) / 255.
            hr_patch = hr_patch.astype(float) / 255.

            data[i * rand_num + j, :, :, 0] = lr_patch
            label[i * rand_num + j, :, :, 0] = hr_patch[6: -6, 6: -6]

    return data, label

def saveToH5File(data, label, savePath):
    with h5py.File(savePath, 'w') as file:
        file.create_dataset('data', data=data)
        file.create_dataset('label', data=label)

if __name__ == "__main__":
    train_data, train_label = preprocess_data(train_path)
    saveToH5File(train_data, train_label, 'train_data.h5')
