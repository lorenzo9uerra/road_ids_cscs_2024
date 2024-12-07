import torch
import torch.nn as nn
import os
import random
import numpy as np


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        device="cuda",
        activation="softmax",
    ):
        super(LSTMModel, self).__init__()
        self.name = "LSTM"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device
        if activation == "softmax":
            self.activation = nn.Softmax(dim=2)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out)
        out = self.activation(out)
        return out


def set_seed(random_seed=42):
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        activation=None,
        name=None,
    ):
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.name = name
        if activation == "relu":
            self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class IncResA(nn.Module):
    def __init__(self, in_channels, name=None):
        super(IncResA, self).__init__()
        self.branch0 = Conv2dBlock(
            in_channels, 32, 1, 1, activation="relu", name=name + "b0"
        )
        self.branch1_1 = Conv2dBlock(
            in_channels, 32, 1, 1, activation="relu", name=name + "b1_1"
        )
        self.branch1_2 = Conv2dBlock(
            32, 32, 3, 1, activation="relu", name=name + "b1_2"
        )
        self.branch2_1 = Conv2dBlock(
            in_channels, 32, 1, 1, activation="relu", name=name + "b2_1"
        )
        self.branch2_2 = Conv2dBlock(
            32, 32, 3, 1, activation="relu", name=name + "b2_2"
        )
        self.branch2_3 = Conv2dBlock(
            32, 32, 3, 1, activation="relu", name=name + "b2_3"
        )
        self.concat = nn.Conv2d(96, 128, 1, 1, padding="same", bias=False)

    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1_1(x)
        branch1 = self.branch1_2(branch1)
        branch2 = self.branch2_1(x)
        branch2 = self.branch2_2(branch2)
        branch2 = self.branch2_3(branch2)
        branches = torch.cat([branch0, branch1, branch2], 1)
        mixed = self.concat(branches)
        return mixed


class IncResB(nn.Module):
    def __init__(self, in_channels, name=None):
        super(IncResB, self).__init__()
        self.branch0 = Conv2dBlock(
            in_channels, 64, 1, 1, activation="relu", name=name + "b0"
        )
        self.branch1_1 = Conv2dBlock(
            in_channels, 64, 1, 1, activation="relu", name=name + "b1_1"
        )
        self.branch1_2 = Conv2dBlock(
            64, 64, (1, 3), 1, activation="relu", name=name + "b1_2"
        )
        self.branch1_3 = Conv2dBlock(
            64, 64, (3, 1), 1, activation="relu", name=name + "b1_3"
        )
        self.concat = nn.Conv2d(128, 448, 1, 1, padding="same", bias=False)

    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1_1(x)
        branch1 = self.branch1_2(branch1)
        branch1 = self.branch1_3(branch1)
        branches = torch.cat([branch0, branch1], 1)
        mixed = self.concat(branches)
        return mixed


class DCNN(nn.Module):
    def __init__(self, random_seed=42):
        super(DCNN, self).__init__()
        self.name = "DCNN"
        # set_seed(random_seed)

        # Stem
        self.conv1 = Conv2dBlock(1, 32, 3, 1, activation="relu", name="conv1")
        self.conv2 = Conv2dBlock(
            32, 32, 3, 1, padding="valid", activation="relu", name="conv2"
        )
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=0)
        self.conv3 = Conv2dBlock(32, 64, 1, 1, activation="relu", name="conv3")
        self.conv4 = Conv2dBlock(64, 128, 3, 1, activation="relu", name="conv4")
        self.conv5 = Conv2dBlock(128, 128, 3, 1, activation="relu", name="conv5")

        # Inception-ResNet-A
        self.incresA = IncResA(128, name="incresA")

        # Reduction-A
        self.red_maxpool_1 = nn.MaxPool2d(3, stride=2, padding=0)
        self.x_red1_c1 = Conv2dBlock(
            128, 192, 3, 2, padding="valid", activation="relu", name="x_red1_c1"
        )
        self.x_red1_c2_1 = Conv2dBlock(
            128, 96, 1, 1, activation="relu", name="x_red1_c2_1"
        )
        self.x_red1_c2_2 = Conv2dBlock(
            96, 96, 3, 1, activation="relu", name="x_red1_c2_2"
        )
        self.x_red1_c2_3 = Conv2dBlock(
            96, 128, 3, 2, padding="valid", activation="relu", name="x_red1_c2_3"
        )

        # Inception-ResNet-B
        self.incresB = IncResB(448, name="incresB")

        # Reduction-B
        self.red_maxpool_2 = nn.MaxPool2d(3, stride=2, padding=0)
        self.x_red2_c11 = Conv2dBlock(
            448, 128, 1, 1, activation="relu", name="x_red2_c11"
        )
        self.x_red2_c12 = Conv2dBlock(
            128, 192, 3, 2, padding="valid", activation="relu", name="x_red2_c12"
        )
        self.x_red2_c21 = Conv2dBlock(
            448, 128, 1, 1, activation="relu", name="x_red2_c21"
        )
        self.x_red2_c22 = Conv2dBlock(
            128, 128, 3, 2, padding="valid", activation="relu", name="x_red2_c22"
        )
        self.x_red2_c31 = Conv2dBlock(
            448, 128, 1, 1, activation="relu", name="x_red2_c31"
        )
        self.x_red2_c32 = Conv2dBlock(
            128, 128, 3, 1, activation="relu", name="x_red2_c32"
        )
        self.x_red2_c33 = Conv2dBlock(
            128, 128, 3, 2, padding="valid", activation="relu", name="x_red2_c33"
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(896, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.incresA(x)
        x_red_11 = self.red_maxpool_1(x)
        x_red_12 = self.x_red1_c1(x)
        x_red_13 = self.x_red1_c2_1(x)
        x_red_13 = self.x_red1_c2_2(x_red_13)
        x_red_13 = self.x_red1_c2_3(x_red_13)
        x = torch.cat([x_red_11, x_red_12, x_red_13], 1)
        x = self.incresB(x)
        x_red_21 = self.red_maxpool_2(x)
        x_red_22 = self.x_red2_c11(x)
        x_red_22 = self.x_red2_c12(x_red_22)
        x_red_23 = self.x_red2_c21(x)
        x_red_23 = self.x_red2_c22(x_red_23)
        x_red_24 = self.x_red2_c31(x)
        x_red_24 = self.x_red2_c32(x_red_24)
        x_red_24 = self.x_red2_c33(x_red_24)
        x = torch.cat([x_red_21, x_red_22, x_red_23, x_red_24], 1)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
