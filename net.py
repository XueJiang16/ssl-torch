import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed1d.pth',
}

dp_rate = 0
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=33, stride=stride,
                     padding=16, bias=False)


class BasicBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn0 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x3(planes, planes*2)

        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(dp_rate)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)


        if self.downsample is not None:
            residual = self.downsample(x)
            # residual = torch.cat((residual,residual),1)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn0 = nn.BatchNorm1d(inplanes)
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=33, bias=False, padding=16)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=65, stride=stride,
                               padding=32, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False, padding=0)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(dp_rate)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.relu(out)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            # residual = torch.cat((residual, residual), 1)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, classification, num_classes=5):
        self.inplanes = 12
        self.classification = classification
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(1, self.inplanes, kernel_size=33, stride=1, padding=16,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(self.inplanes, self.inplanes, kernel_size=33, stride=2, padding=16,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(self.inplanes)
        self.downsample = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(self.inplanes, self.inplanes, kernel_size=33, stride=1, padding=16,
                               bias=False)
        self.dropout = nn.Dropout(dp_rate)
        self.layer1 = self._make_layer(block, 12, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 24, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 48, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 96, layers[3], stride=2)
        # self.layer5 = self._make_layer(block, self.inplanes, layers[4], stride=2)
        self.bn_final = nn.BatchNorm1d(96*2)
        self.avgpool = nn.AdaptiveAvgPool1d(2)
        self.fc1 = nn.Linear(96*4, 384)
        self.bn3 = nn.BatchNorm1d(384)
        self.fc2 = nn.Linear(384, 192)
        self.bn4 = nn.BatchNorm1d(192)
        self.fc3 = nn.Linear(192, 5)
        self.softmax = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv3(out)
        residual = self.downsample(x)
        out += residual
        x = self.relu(out)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.layer5(x)
        x = self.bn_final(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.classification:
            x = self.fc1(x)
            x = self.bn3(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.bn4(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc3(x)
            # x = self.softmax(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [ 2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

