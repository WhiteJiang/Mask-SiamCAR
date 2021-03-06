import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
from models.features import Feature

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_channels, out_channels, stride=1):
    """
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param stride: 步长
    :return:
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """"
    for resnet18 resnet34
    """
    expansion = 1

    def __int__(self, in_channels, channels, stride=1, downsample=None):
        super(BasicBlock, self).__int__()
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(Feature):
    """
    for feature extract
    """
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__int__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        padding = 2 - stride
        assert stride == 1 or dilation == 1, "stride and dilation must have one equals to zero at least"
        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=padding, bias=False,
                               dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out


class Bottleneck_nop(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super(Bottleneck_nop, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        s = residual.size(3)
        residual = residual[:, :, 1:s - 1, 1:s - 1]

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, layer4=False, layer3=False):
        self.in_channels = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)

        self.feature_size = 128 * block.expansion

        if layer3:
            self.layer3 = self.make_layer(block, 256, layers[2], stride=1, dilation=2)
            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x: x

        if layer4:
            self.layer4 = self.make_layer(block, 512, layers[3], stride=1, dilation=4)
            self.feature_size = 512 * block.expansion
        else:
            self.layer4 = lambda x: x
        # parm  init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_layer(self, block, channels, blocks, stride=1, dilation=1):
        downsample = None
        dd = dilation
        if stride != 1 or self.in_channels != channels * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, channels * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(channels * block.expansion),
                )
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = dd
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, channels * block.expansion,
                              kernel_size=1, stride=stride, bias=False,
                              padding=padding, dilation=dd),
                    nn.BatchNorm2d(channels * block.expansion)
                )
        layers = [block(self.in_channels, channels, stride, downsample, dilation=dd)]
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels, dilation=dd))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        p0 = self.relu(x)
        x = self.maxpool(p0)

        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)

        return p0, p1, p2, p3


class ResAdjust(nn.Module):

    def __init__(self, block=Bottleneck, out_channels=256, adjust_number=1, fuse_layers=None):
        if fuse_layers is None:
            fuse_layers = [2, 3, 4]
        super(ResAdjust, self).__init__()
        self.fuse_layers = set(fuse_layers)

        if 2 in self.fuse_layers:
            self.layer2 = self.make_layer(block, 128, 1, out_channels, adjust_number)
        if 3 in self.fuse_layers:
            self.layer3 = self.make_layer(block, 256, 2, out_channels, adjust_number)
        if 4 in self.fuse_layers:
            self.layer4 = self.make_layer(block, 512, 4, out_channels, adjust_number)

    def make_layer(self, block, channels, dilation, out, number=1):
        layers = []
        for _ in range(number):
            layer = block(channels * block.expansion, channels, dilation=dilation)
            layers.append(layer)

        downsample = nn.Sequential(
            nn.Conv2d(channels * block.expansion, out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out)
        )
        layers.append(downsample)
        return nn.Sequential(*layers)

    def forward(self, p2, p3, p4):
        outputs = []

        if 2 in self.fuse_layers:
            outputs.append(self.layer2(p2))
        if 3 in self.fuse_layers:
            outputs.append(self.layer3(p3))
        if 4 in self.fuse_layers:
            outputs.append(self.layer4(p4))
        # return torch.cat(outputs,1)

        return outputs


def resnet50(pretarined=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretarined:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    return model


if __name__ == '__main__':
    net = resnet50()
    print(net)
    net = net.cuda()

    var = torch.FloatTensor(1, 3, 127, 127).cuda()
    var = Variable(var)

    net(var)
    print('*********')
    var = torch.FloatTensor(1, 3, 255, 255).cuda()
    var = Variable(var)

    net(var)
