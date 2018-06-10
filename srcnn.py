import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import h5py
import numpy as np
import cv2

data = torch.set_default_tensor_type('torch.DoubleTensor')

def read_data(file):
    with h5py.File(file, 'r') as h:
        data = np.array(h.get('data'))
        label = np.array(h.get('label'))
        data = np.transpose(data, (0, 3, 1, 2))
        label = np.transpose(label, (0, 3, 1, 2))
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        return data, label

EPOCH = 1
BATCH_SIZE = 100
LR = 0.001

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 9),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 1, 5),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        output = self.conv3(x)
        return output

cnn = CNN()

if torch.cuda.is_available():
    cnn = cnn.cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

# training
def train():
    data, label = read_data('./train_data.h5')
    dataset = Data.TensorDataset(data_tensor=data, target_tensor=label)
    train_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    cnn.load_state_dict(torch.load('cnn_params.pkl'))

    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x)
            b_y = Variable(y)

            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])

    torch.save(cnn, 'cnn.pkl')
    torch.save(cnn.state_dict(), 'cnn_params.pkl')

# testing
def test(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    height, width, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    y, _, _ = cv2.split(img)

    lr_img = cv2.resize(y, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
    lr_img = cv2.resize(lr_img, (width, height), interpolation=cv2.INTER_CUBIC)

    bicubic_img = cv2.resize(img, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
    bicubic_img = cv2.resize(bicubic_img, (width, height), interpolation=cv2.INTER_CUBIC)
    bicubic_img = cv2.cvtColor(bicubic_img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite('bicubic.bmp', bicubic_img)

    data = np.zeros((1, height, width, 1), dtype=np.double)
    label = np.zeros((1, height - 12, width - 12, 1), dtype=np.double)
    y = y.astype(float) / 255.
    lr_img = lr_img.astype(float) / 255.
    data[0, :, :, 0] = lr_img
    label[0, :, :, 0] = y[6: -6, 6: -6]
    data = np.transpose(data, (0, 3, 1, 2))
    label = np.transpose(label, (0, 3, 1, 2))
    data = torch.from_numpy(data)
    label = torch.from_numpy(label)
    if torch.cuda.is_available():
        data = data.cuda()
        label = label.cuda()

    dataset = Data.TensorDataset(data_tensor=data, target_tensor=label)
    test_loader = Data.DataLoader(dataset=dataset, batch_size=1)

    cnn.load_state_dict(torch.load('cnn_params.pkl'))

    for step, (x, y) in enumerate(test_loader):
        b_x = Variable(x)
        b_y = Variable(y)

        output = cnn(b_x)
        loss = loss_func(output, b_y)
        print('loss: %.4f' % loss.data[0])

    output = output * 255
    if torch.cuda.is_available():
        output = output.cpu()
    output = output.data.numpy()
    output[output[:] > 255] = 255
    output[output[:] < 0] = 0
    img[6: -6, 6: -6, 0] = output[0, 0, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite('output.bmp', img)

if __name__ == '__main__':
    train()
    test('./dataset/Test/Set5/butterfly_GT.bmp')
