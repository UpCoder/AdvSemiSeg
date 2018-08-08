import tensorflow as tf
from tensorflow.contrib import slim

class Classifier_Module:
    @slim.add_arg_scope
    def __init__(self, input_tensor, dilation_series, padding_series, num_classes):
        def pad_conv(x, padding, dilation, num_classes):
            paddings = tf.constant(
                [[0, 0], [padding, padding ], [padding, padding], [0, 0]])
            out = tf.pad(x, paddings=paddings)
            out = slim.conv2d(out, num_classes, kernel_size=3, stride=1, padding='VALID', rate=dilation)
            return out
        out = pad_conv(input_tensor, padding_series[0], dilation_series[0], num_classes)
        for idx, (dilation, padding) in enumerate(zip(dilation_series, padding_series)):
            if idx == 0:
                continue
            out += pad_conv(input_tensor, padding, dilation, num_classes)
        self.out = out

class Bottleneck:
    expansion = 4
    @slim.add_arg_scope
    def __init__(self, input_tensor, planes, stride=1, dilation=1, downsample=None):
        self.residual = input_tensor
        self.conv1 = slim.conv2d(input_tensor, planes, kernel_size=1, stride=stride, biases_initializer=None)
        self.bn1 = slim.batch_norm(self.conv1)
        self.relu1 = tf.nn.relu(self.bn1)

        self.conv2 = slim.conv2d(self.relu1, planes, kernel_size=3, stride=1, padding='SAME',
                                 biases_initializer=None, rate=dilation)
        self.bn2 = slim.batch_norm(self.conv2)
        self.relu2 = tf.nn.relu(self.bn2)

        self.conv3 = slim.conv2d(self.relu2, planes * Bottleneck.expansion, kernel_size=1, stride=1, padding='SAME',
                                 biases_initializer=None)
        self.bn3 = slim.batch_norm(self.conv3)

        self.downsample = downsample
        self.stride = stride
        if self.downsample is not None:
            self.residual = self.downsample(input_tensor)
        self.bn3 += self.residual

        self.relu3 = tf.nn.relu(self.bn3)
        self.out = self.relu3

class ResNet:
    @slim.add_arg_scope
    def _make_pred(self, input_tensor, dilation_series, padding_series, num_classes):
        return Classifier_Module(input_tensor, dilation_series=dilation_series,
                                 padding_series=padding_series, num_classes=num_classes).out

    @slim.add_arg_scope
    def _make_layer(self, input_tensor, block, planes, blocks, stride=1, dilation=1):
        def sequential(input_tensor):
            conv = slim.conv2d(input_tensor, planes * block.expansion, kernel_size=1, stride=stride,
                               biases_initializer=None)
            bn = slim.batch_norm(conv)
            return bn

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = sequential
        output = block(input_tensor,planes,stride=stride, dilation=dilation, downsample=downsample).out

        # self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            output = block(
                output, planes, dilation=dilation
            ).out
        return output

    def __init__(self, input_tensor, block, layers, num_classes, is_training):
        self.out = None
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                self.inplanes = 64
                self.conv1 = slim.conv2d(input_tensor, 64, kernel_size=7, stride=2, padding='SAME', biases_initializer=None)
                self.bn1 = slim.batch_norm(self.conv1)
                self.relu1 = tf.nn.relu(self.bn1)
                self.maxpool = slim.max_pool2d(self.relu1, kernel_size=3, stride=2, padding='SAME')

                self.layer1 = self._make_layer(self.maxpool, block, 64, layers[0])
                self.layer2 = self._make_layer(self.layer1, block, 128, layers[1], stride=2)
                self.layer3 = self._make_layer(self.layer2, block, 256, layers[2], stride=1, dilation=2)
                self.layer4 = self._make_layer(self.layer3, block, 512, layers[3], stride=1, dilation=4)
                print(self.layer4)
                self.layer5 = self._make_pred(self.layer4, [6,12,18,24],[6,12,18,24], num_classes)
                print('final layer is ', self.layer5)
                self.out = self.layer5


if __name__ == '__main__':
    input_tensor = tf.placeholder(tf.float32, [None, 256, 256, 1], name='x_input')
    is_training = tf.placeholder(tf.bool, [], name='is_training')
    resnet = ResNet(input_tensor, Bottleneck, [3, 4, 23, 3], num_classes=2, is_training=is_training)


