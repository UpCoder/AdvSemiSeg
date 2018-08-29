# -*- coding=utf-8 -*-
import torch.nn as nn
import math
from torch.autograd import Variable
import torch
import numpy as np
from packaging import version
from model.ConvLSTM import ConvLSTMCell
affine_par = True


def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
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


class UpSample_Module(nn.Module):
    def __init__(self, inplanes, planes, output_size, stride=1):
        super(UpSample_Module, self).__init__()
        self.inplanes = inplanes
        if version.parse(torch.__version__) >= version.parse('0.4.0'):
            self.interp = nn.Upsample(size=(output_size, output_size), mode='bilinear', align_corners=True)
        else:
            self.interp = nn.Upsample(size=(output_size, output_size), mode='bilinear')
        self.conv = nn.Conv2d(self.inplanes, planes, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.interp(x)
        out = self.conv(out)
        out = self.bn1(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
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

class Classifier_Module(nn.Module):
    def __init__(self, dilation_series, padding_series, input_channel, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(input_channel, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation,
                          bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
            return out



class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, slice_num, gpu_id=None):
        self.inplanes = 64
        zoom_ratio = 4
        extract_ratio = 1
        self.growth_rate = 60
        self.slice_num = slice_num
        self.gpu_id = gpu_id
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, self.growth_rate, layers[0])
        # print(self.layer1)
        self.layer2 = self._make_layer(block, self.growth_rate * 2, layers[1], stride=2)
        # print(self.layer2)
        self.layer3 = self._make_layer(block, self.growth_rate * 4, layers[2], stride=1, dilation=2)

        # print(self.layer3)
        self.layer4 = self._make_layer(block, self.growth_rate * 8, layers[3], stride=1, dilation=4)
        # print(self.layer4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24],
                                            int(self.growth_rate * 2 / extract_ratio), num_classes)
        # print(self.layer5)


        self.lstmCell_layer1 = ConvLSTMCell(self.growth_rate * 4, hidden_size= self.growth_rate * 4 // zoom_ratio, gpu_id=self.gpu_id)
        self.conv1_layer1 = nn.Conv2d(self.growth_rate * 4 // zoom_ratio * self.slice_num,
                                      int(self.growth_rate * 4 / extract_ratio), kernel_size=1,
                                      stride=1, padding=0, bias=False)

        self.lstmCell_layer2 = ConvLSTMCell(self.growth_rate * 8, hidden_size=self.growth_rate * 8 // zoom_ratio,
                                            gpu_id=self.gpu_id)
        self.conv2_layer2 = nn.Conv2d(self.growth_rate * 8 // zoom_ratio * self.slice_num,
                                      int(self.growth_rate * 8 / extract_ratio), kernel_size=1,
                                      stride=1, padding=0, bias=False)

        self.lstmCell_layer3 = ConvLSTMCell(self.growth_rate * 16, hidden_size=self.growth_rate * 16 // zoom_ratio,
                                            gpu_id=self.gpu_id)
        self.conv3_layer3 = nn.Conv2d(self.growth_rate * 16 // zoom_ratio * self.slice_num,
                                      int(self.growth_rate * 16 / extract_ratio), kernel_size=1, stride=1, padding=0,
                                      bias=False)

        self.lstmCell_layer4 = ConvLSTMCell(self.growth_rate * 32, hidden_size=self.growth_rate * 32 // zoom_ratio,
                                            gpu_id=self.gpu_id)
        self.conv4_layer4 = nn.Conv2d(self.growth_rate * 32 // zoom_ratio * self.slice_num,
                                      int(self.growth_rate * 32 / extract_ratio), kernel_size=1, stride=1, padding=0,
                                      bias=False)

        self.layer4_upsample = UpSample_Module(int(self.growth_rate * 32 / extract_ratio),
                                               int(self.growth_rate * 16 / extract_ratio), 51, stride=1)
        self.layer4_upsample_conv = nn.Conv2d(int(self.growth_rate * 32 / extract_ratio),
                                              int(self.growth_rate * 16 / extract_ratio), kernel_size=1, stride=1,
                                              padding=0, bias=False)

        self.layer3_upsample = UpSample_Module(int(self.growth_rate * 16 / extract_ratio),
                                               int(self.growth_rate * 8 / extract_ratio), 51, stride=1)
        self.layer3_upsample_conv = nn.Conv2d(int(self.growth_rate * 16 / extract_ratio),
                                              int(self.growth_rate * 8 / extract_ratio), kernel_size=1, stride=1,
                                              padding=0, bias=False)

        self.layer2_upsample = UpSample_Module(int(self.growth_rate * 8 / extract_ratio),
                                               int(self.growth_rate * 4 / extract_ratio), 101, stride=1)
        self.layer2_upsample_conv = nn.Conv2d(int(self.growth_rate * 8 / extract_ratio),
                                              int(self.growth_rate * 4 / extract_ratio), kernel_size=1, stride=1,
                                              padding=0, bias=False)

        self.layer1_upsample = UpSample_Module(int(self.growth_rate * 4 / extract_ratio),
                                               int(self.growth_rate * 2 / extract_ratio), 200, stride=1)
        self.layer1_upsample_conv = nn.Conv2d(int(self.growth_rate * 4 / extract_ratio),
                                              int(self.growth_rate * 2 / extract_ratio), kernel_size=1, stride=1,
                                              padding=0, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self,block, dilation_series, padding_series,input_channel, num_classes):
        return block(dilation_series, padding_series, input_channel, num_classes)

    def forward(self, x):
        splits = torch.split(x, split_size_or_sections=1, dim=1)
        layer4_state = None
        layer3_state = None
        layer2_state = None
        layer1_state = None
        layer4s = []
        layer3s = []
        layer2s = []
        layer1s = []
        for split in splits:
            x = split
            # print(x.size())
            x = self.conv1(x)
            print('after conv1 shape is ', x.size())
            x = self.bn1(x)
            x = self.relu(x)
            layer1 = self.layer1(x)
            layer1 = self.maxpool(layer1)
            print('after maxpool shape is ', layer1.size())

            layer1s.append(layer1)
            # print('layer1: ', list(layer1.size()))
            layer2 = self.layer2(layer1)
            layer2s.append(layer2)
            # print('layer2: ', list(layer2.size()))
            layer3 = self.layer3(layer2)
            layer3s.append(layer3)
            # print('layer3: ', list(layer3.size()))
            layer4 = self.layer4(layer3)
            layer4s.append(layer4)
            # print('layer4: ', list(layer4.size()))
        print('layer4s is ', len(layer4s))
        for idx in range(len(layer4s)):
            print('\t', layer4s[idx].size())

        print('layer3s is ', len(layer3s))
        for idx in range(len(layer3s)):
            print('\t', layer3s[idx].size())

        print('layer2s is ', len(layer2s))
        for idx in range(len(layer2s)):
            print('\t', layer2s[idx].size())

        print('layer1s is ', len(layer1s))
        for idx in range(len(layer1s)):
            print('\t', layer1s[idx].size())

        layer4_outputs = []
        layer3_outputs = []
        layer2_outputs = []
        layer1_outputs = []
        for idx in range(len(splits)):
            layer4_state = self.lstmCell_layer4(layer4s[idx], layer4_state)
            # print('layer4_shate shape is ', layer4_state[0].size())
            layer4_outputs.append(layer4_state[0])

            layer3_state = self.lstmCell_layer3(layer3s[idx], layer3_state)
            layer3_outputs.append(layer3_state[0])

            layer2_state = self.lstmCell_layer2(layer2s[idx], layer2_state)
            layer2_outputs.append(layer2_state[0])

            layer1_state = self.lstmCell_layer1(layer1s[idx], layer1_state)
            layer1_outputs.append(layer1_state[0])

        layer4_output = torch.cat(layer4_outputs, 1)
        # print('layer4_output shape: ', layer4_output.size())
        layer4 = self.conv4_layer4(layer4_output)
        layer3_output = torch.cat(layer3_outputs, 1)
        layer3 = self.conv3_layer3(layer3_output)
        layer2_output = torch.cat(layer2_outputs, 1)
        layer2 = self.conv2_layer2(layer2_output)
        layer1_output = torch.cat(layer1_outputs, 1)
        layer1 = self.conv1_layer1(layer1_output)

        layer4_upsample = self.layer4_upsample(layer4)
        # print('The size of layer4 is ', layer4_upsample.size())
        # print('The size of layer3 is ', layer3.size())

        # print('before concat, upsample size: ', layer4_upsample.size(), ' layer3 size is: ', layer3.size())
        layer3_merged = torch.cat((layer4_upsample, layer3), 1)
        layer3_output = self.layer4_upsample_conv(layer3_merged)
        layer3_upsample = self.layer3_upsample(layer3_output)

        # print('before concat, upsample size: ', layer3_upsample.size(), ' layer2 size is: ', layer2.size())
        layer2_merged = torch.cat((layer3_upsample, layer2), 1)
        layer2_output = self.layer3_upsample_conv(layer2_merged)
        layer2_upsample = self.layer2_upsample(layer2_output)

        # print('before concat, upsample size: ', layer2_upsample.size(), ' layer1 size is: ', layer1.size())
        layer1_merged = torch.cat((layer2_upsample, layer1), 1)
        layer1_output = self.layer2_upsample_conv(layer1_merged)
        layer1_upsample = self.layer1_upsample(layer1_output)

        print('layer1_upsample shape: ', layer1_upsample.size())
        layer_5 = self.layer5(layer1_upsample)
        # print('layer5: ', list(layer_5.size()))

        return layer_5

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for 
        the last classification layer. Note that for each batchnorm layer, 
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)
    
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj+=1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i


    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10*args.learning_rate}] 


def Res_Deeplab(num_classes=21, slice_num=5, gpu_id=None):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, slice_num=slice_num, gpu_id=gpu_id)
    return model


if __name__ == '__main__':
    slice_num = 5
    gpu_id = 0
    res_deeplab = Res_Deeplab(num_classes=2, slice_num=slice_num, gpu_id=gpu_id)
    res_deeplab.cuda(gpu_id)
    input_tensor = Variable(torch.Tensor(np.random.random([1, slice_num, 400, 400]))).cuda(gpu_id)
    output_tensor = res_deeplab(input_tensor).data.cpu().numpy()
    print(np.shape(output_tensor))

    # import numpy as np
    # # 我们我们使用同一个conv, 那么我们就是共享参数的
    # # conv = nn.Conv2d(1, 3, kernel_size=3, stride=1, bias=False)
    # data = np.random.random([1, 1, 200, 200])
    # data_tensor = torch.Tensor(data)
    # for _ in range(3):
    #     conv = nn.Conv2d(1, 3, kernel_size=3, stride=1, bias=False)
    #     output = conv(data_tensor).data.cpu().numpy()
    #     print('Sum of output is ', np.sum(output))
