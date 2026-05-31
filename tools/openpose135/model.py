"""PyTorch nn.Modules for CMU OpenPose: BODY_25 + hand + face (71-channel).

Architectures match CMU's original Caffe prototxts so weights converted by
caffemodel2pytorch (and rehosted as .pth) load cleanly via util.transfer.

Sources:
- BODY_25 (bodypose_25_model): https://github.com/TracelessLe/OpenPose.PyTorch
- hand (handpose_model)      : https://github.com/Hzzone/pytorch-openpose
- face (FaceNet)             : https://github.com/lllyasviel/ControlNet-v1-1-nightly
"""
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Module, ReLU, init


def make_layers(block, no_relu_layers, prelu_layers=()):
    layers = []
    for layer_name, v in block.items():
        if "pool" in layer_name:
            layers.append((layer_name, nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])))
        else:
            conv = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
            layers.append((layer_name, conv))
            if layer_name not in no_relu_layers:
                if layer_name in prelu_layers:
                    layers.append(("prelu" + layer_name[4:], nn.PReLU(v[1])))
                else:
                    layers.append(("relu_" + layer_name, nn.ReLU(inplace=True)))
    return nn.Sequential(OrderedDict(layers))


def make_layers_mconv(block, no_relu_layers):
    modules = []
    for layer_name, v in block.items():
        layers = []
        if "pool" in layer_name:
            layers.append((layer_name, nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])))
        else:
            conv = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
            layers.append((layer_name, conv))
            if layer_name not in no_relu_layers:
                layers.append(("Mprelu" + layer_name[5:], nn.PReLU(v[1])))
        modules.append(nn.Sequential(OrderedDict(layers)))
    return nn.ModuleList(modules)


class bodypose_25_model(nn.Module):
    def __init__(self):
        super().__init__()
        no_relu_layers = [
            "Mconv7_stage0_L1", "Mconv7_stage0_L2",
            "Mconv7_stage1_L1", "Mconv7_stage1_L2",
            "Mconv7_stage2_L2", "Mconv7_stage3_L2",
        ]
        prelu_layers = ["conv4_2", "conv4_3_CPM", "conv4_4_CPM"]

        block0 = OrderedDict([
            ("conv1_1", [3, 64, 3, 1, 1]),
            ("conv1_2", [64, 64, 3, 1, 1]),
            ("pool1_stage1", [2, 2, 0]),
            ("conv2_1", [64, 128, 3, 1, 1]),
            ("conv2_2", [128, 128, 3, 1, 1]),
            ("pool2_stage1", [2, 2, 0]),
            ("conv3_1", [128, 256, 3, 1, 1]),
            ("conv3_2", [256, 256, 3, 1, 1]),
            ("conv3_3", [256, 256, 3, 1, 1]),
            ("conv3_4", [256, 256, 3, 1, 1]),
            ("pool3_stage1", [2, 2, 0]),
            ("conv4_1", [256, 512, 3, 1, 1]),
            ("conv4_2", [512, 512, 3, 1, 1]),
            ("conv4_3_CPM", [512, 256, 3, 1, 1]),
            ("conv4_4_CPM", [256, 128, 3, 1, 1]),
        ])
        self.model0 = make_layers(block0, no_relu_layers, prelu_layers)

        blocks = {}

        # L2 (PAFs) — stage0
        blocks["Mconv1_stage0_L2"] = OrderedDict([
            ("Mconv1_stage0_L2_0", [128, 96, 3, 1, 1]),
            ("Mconv1_stage0_L2_1", [96, 96, 3, 1, 1]),
            ("Mconv1_stage0_L2_2", [96, 96, 3, 1, 1]),
        ])
        for i in range(2, 6):
            blocks[f"Mconv{i}_stage0_L2"] = OrderedDict([
                (f"Mconv{i}_stage0_L2_0", [288, 96, 3, 1, 1]),
                (f"Mconv{i}_stage0_L2_1", [96, 96, 3, 1, 1]),
                (f"Mconv{i}_stage0_L2_2", [96, 96, 3, 1, 1]),
            ])
        blocks["Mconv6_7_stage0_L2"] = OrderedDict([
            ("Mconv6_stage0_L2", [288, 256, 1, 1, 0]),
            ("Mconv7_stage0_L2", [256, 52, 1, 1, 0]),
        ])

        # L2 (PAFs) — stages 1..3
        for s in range(1, 4):
            blocks[f"Mconv1_stage{s}_L2"] = OrderedDict([
                (f"Mconv1_stage{s}_L2_0", [180, 128, 3, 1, 1]),
                (f"Mconv1_stage{s}_L2_1", [128, 128, 3, 1, 1]),
                (f"Mconv1_stage{s}_L2_2", [128, 128, 3, 1, 1]),
            ])
            for i in range(2, 6):
                blocks[f"Mconv{i}_stage{s}_L2"] = OrderedDict([
                    (f"Mconv{i}_stage{s}_L2_0", [384, 128, 3, 1, 1]),
                    (f"Mconv{i}_stage{s}_L2_1", [128, 128, 3, 1, 1]),
                    (f"Mconv{i}_stage{s}_L2_2", [128, 128, 3, 1, 1]),
                ])
            blocks[f"Mconv6_7_stage{s}_L2"] = OrderedDict([
                (f"Mconv6_stage{s}_L2", [384, 512, 1, 1, 0]),
                (f"Mconv7_stage{s}_L2", [512, 52, 1, 1, 0]),
            ])

        # L1 (heatmaps) — stage0
        blocks["Mconv1_stage0_L1"] = OrderedDict([
            ("Mconv1_stage0_L1_0", [180, 96, 3, 1, 1]),
            ("Mconv1_stage0_L1_1", [96, 96, 3, 1, 1]),
            ("Mconv1_stage0_L1_2", [96, 96, 3, 1, 1]),
        ])
        for i in range(2, 6):
            blocks[f"Mconv{i}_stage0_L1"] = OrderedDict([
                (f"Mconv{i}_stage0_L1_0", [288, 96, 3, 1, 1]),
                (f"Mconv{i}_stage0_L1_1", [96, 96, 3, 1, 1]),
                (f"Mconv{i}_stage0_L1_2", [96, 96, 3, 1, 1]),
            ])
        blocks["Mconv6_7_stage0_L1"] = OrderedDict([
            ("Mconv6_stage0_L1", [288, 256, 1, 1, 0]),
            ("Mconv7_stage0_L1", [256, 26, 1, 1, 0]),
        ])

        # L1 (heatmaps) — stage1
        blocks["Mconv1_stage1_L1"] = OrderedDict([
            ("Mconv1_stage1_L1_0", [206, 128, 3, 1, 1]),
            ("Mconv1_stage1_L1_1", [128, 128, 3, 1, 1]),
            ("Mconv1_stage1_L1_2", [128, 128, 3, 1, 1]),
        ])
        for i in range(2, 6):
            blocks[f"Mconv{i}_stage1_L1"] = OrderedDict([
                (f"Mconv{i}_stage1_L1_0", [384, 128, 3, 1, 1]),
                (f"Mconv{i}_stage1_L1_1", [128, 128, 3, 1, 1]),
                (f"Mconv{i}_stage1_L1_2", [128, 128, 3, 1, 1]),
            ])
        blocks["Mconv6_7_stage1_L1"] = OrderedDict([
            ("Mconv6_stage1_L1", [384, 512, 1, 1, 0]),
            ("Mconv7_stage1_L1", [512, 26, 1, 1, 0]),
        ])

        for k in blocks:
            blocks[k] = make_layers_mconv(blocks[k], no_relu_layers)
        self.models = nn.ModuleDict(blocks)

    def _mconv_forward(self, x, models):
        outs = []
        out = x
        for m in models:
            out = m(out)
            outs.append(out)
        return torch.cat(outs, 1)

    def forward(self, x):
        out0 = self.model0(x)
        # L2 (PAFs) — 4 stages
        tout = out0
        for s in range(4):
            tout = self._mconv_forward(tout, self.models[f"Mconv1_stage{s}_L2"])
            for v in range(2, 6):
                tout = self._mconv_forward(tout, self.models[f"Mconv{v}_stage{s}_L2"])
            tout = self.models[f"Mconv6_7_stage{s}_L2"][0](tout)
            tout = self.models[f"Mconv6_7_stage{s}_L2"][1](tout)
            outL2 = tout
            tout = torch.cat([out0, tout], 1)
        # L1 (heatmaps) — stage0
        tout = self._mconv_forward(tout, self.models["Mconv1_stage0_L1"])
        for v in range(2, 6):
            tout = self._mconv_forward(tout, self.models[f"Mconv{v}_stage0_L1"])
        tout = self.models["Mconv6_7_stage0_L1"][0](tout)
        tout = self.models["Mconv6_7_stage0_L1"][1](tout)
        outS0L1 = tout
        tout = torch.cat([out0, outS0L1, outL2], 1)
        # L1 (heatmaps) — stage1
        tout = self._mconv_forward(tout, self.models["Mconv1_stage1_L1"])
        for v in range(2, 6):
            tout = self._mconv_forward(tout, self.models[f"Mconv{v}_stage1_L1"])
        tout = self.models["Mconv6_7_stage1_L1"][0](tout)
        outS1L1 = self.models["Mconv6_7_stage1_L1"][1](tout)
        return outL2, outS1L1


class handpose_model(nn.Module):
    def __init__(self):
        super().__init__()
        no_relu_layers = [
            "conv6_2_CPM",
            "Mconv7_stage2", "Mconv7_stage3", "Mconv7_stage4", "Mconv7_stage5", "Mconv7_stage6",
        ]

        block1_0 = OrderedDict([
            ("conv1_1", [3, 64, 3, 1, 1]),
            ("conv1_2", [64, 64, 3, 1, 1]),
            ("pool1_stage1", [2, 2, 0]),
            ("conv2_1", [64, 128, 3, 1, 1]),
            ("conv2_2", [128, 128, 3, 1, 1]),
            ("pool2_stage1", [2, 2, 0]),
            ("conv3_1", [128, 256, 3, 1, 1]),
            ("conv3_2", [256, 256, 3, 1, 1]),
            ("conv3_3", [256, 256, 3, 1, 1]),
            ("conv3_4", [256, 256, 3, 1, 1]),
            ("pool3_stage1", [2, 2, 0]),
            ("conv4_1", [256, 512, 3, 1, 1]),
            ("conv4_2", [512, 512, 3, 1, 1]),
            ("conv4_3", [512, 512, 3, 1, 1]),
            ("conv4_4", [512, 512, 3, 1, 1]),
            ("conv5_1", [512, 512, 3, 1, 1]),
            ("conv5_2", [512, 512, 3, 1, 1]),
            ("conv5_3_CPM", [512, 128, 3, 1, 1]),
        ])
        block1_1 = OrderedDict([
            ("conv6_1_CPM", [128, 512, 1, 1, 0]),
            ("conv6_2_CPM", [512, 22, 1, 1, 0]),
        ])

        blocks = {"block1_0": block1_0, "block1_1": block1_1}
        for i in range(2, 7):
            blocks[f"block{i}"] = OrderedDict([
                (f"Mconv1_stage{i}", [150, 128, 7, 1, 3]),
                (f"Mconv2_stage{i}", [128, 128, 7, 1, 3]),
                (f"Mconv3_stage{i}", [128, 128, 7, 1, 3]),
                (f"Mconv4_stage{i}", [128, 128, 7, 1, 3]),
                (f"Mconv5_stage{i}", [128, 128, 7, 1, 3]),
                (f"Mconv6_stage{i}", [128, 128, 1, 1, 0]),
                (f"Mconv7_stage{i}", [128, 22, 1, 1, 0]),
            ])
        for k in blocks:
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_0 = blocks["block1_0"]
        self.model1_1 = blocks["block1_1"]
        self.model2 = blocks["block2"]
        self.model3 = blocks["block3"]
        self.model4 = blocks["block4"]
        self.model5 = blocks["block5"]
        self.model6 = blocks["block6"]

    def forward(self, x):
        out1_0 = self.model1_0(x)
        out1_1 = self.model1_1(out1_0)
        x = torch.cat([out1_1, out1_0], 1)
        x = self.model2(x)
        x = torch.cat([x, out1_0], 1)
        x = self.model3(x)
        x = torch.cat([x, out1_0], 1)
        x = self.model4(x)
        x = torch.cat([x, out1_0], 1)
        x = self.model5(x)
        x = torch.cat([x, out1_0], 1)
        x = self.model6(x)
        return x


class FaceNet(Module):
    """71-channel cascading-heatmap face net matching CMU openpose face_deploy.prototxt."""

    def __init__(self):
        super().__init__()
        self.relu = ReLU()
        self.max_pooling_2d = MaxPool2d(kernel_size=2, stride=2)
        self.conv1_1 = Conv2d(3, 64, 3, 1, 1)
        self.conv1_2 = Conv2d(64, 64, 3, 1, 1)
        self.conv2_1 = Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = Conv2d(128, 128, 3, 1, 1)
        self.conv3_1 = Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = Conv2d(256, 256, 3, 1, 1)
        self.conv3_4 = Conv2d(256, 256, 3, 1, 1)
        self.conv4_1 = Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = Conv2d(512, 512, 3, 1, 1)
        self.conv4_4 = Conv2d(512, 512, 3, 1, 1)
        self.conv5_1 = Conv2d(512, 512, 3, 1, 1)
        self.conv5_2 = Conv2d(512, 512, 3, 1, 1)
        self.conv5_3_CPM = Conv2d(512, 128, 3, 1, 1)

        self.conv6_1_CPM = Conv2d(128, 512, 1, 1, 0)
        self.conv6_2_CPM = Conv2d(512, 71, 1, 1, 0)

        for stage in range(2, 7):
            setattr(self, f"Mconv1_stage{stage}", Conv2d(199, 128, 7, 1, 3))
            for k in range(2, 6):
                setattr(self, f"Mconv{k}_stage{stage}", Conv2d(128, 128, 7, 1, 3))
            setattr(self, f"Mconv6_stage{stage}", Conv2d(128, 128, 1, 1, 0))
            setattr(self, f"Mconv7_stage{stage}", Conv2d(128, 71, 1, 1, 0))

        for m in self.modules():
            if isinstance(m, Conv2d):
                init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.relu(self.conv1_1(x))
        h = self.relu(self.conv1_2(h))
        h = self.max_pooling_2d(h)
        h = self.relu(self.conv2_1(h))
        h = self.relu(self.conv2_2(h))
        h = self.max_pooling_2d(h)
        h = self.relu(self.conv3_1(h))
        h = self.relu(self.conv3_2(h))
        h = self.relu(self.conv3_3(h))
        h = self.relu(self.conv3_4(h))
        h = self.max_pooling_2d(h)
        h = self.relu(self.conv4_1(h))
        h = self.relu(self.conv4_2(h))
        h = self.relu(self.conv4_3(h))
        h = self.relu(self.conv4_4(h))
        h = self.relu(self.conv5_1(h))
        h = self.relu(self.conv5_2(h))
        h = self.relu(self.conv5_3_CPM(h))
        feature_map = h

        h = self.relu(self.conv6_1_CPM(h))
        h = self.conv6_2_CPM(h)
        heatmaps = [h]
        for stage in range(2, 7):
            h = torch.cat([h, feature_map], dim=1)
            for k in range(1, 6):
                h = self.relu(getattr(self, f"Mconv{k}_stage{stage}")(h))
            h = self.relu(getattr(self, f"Mconv6_stage{stage}")(h))
            h = getattr(self, f"Mconv7_stage{stage}")(h)
            heatmaps.append(h)
        return heatmaps
