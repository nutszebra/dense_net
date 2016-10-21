import six
import numpy as np
import chainer.links as L
import chainer.functions as F
import trainer.nutszebra_chainer as nutszebra_chainer
import functools
from collections import defaultdict


class BN_ReLU_Conv(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(BN_ReLU_Conv, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel, filter_size, stride, pad),
            bn=L.BatchNormalization(in_channel),
        )

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def __call__(self, x, train=False):
        return self.conv(F.relu(self.bn(x, test=not train)))

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)


class Transition_Layer(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel):
        super(Transition_Layer, self).__init__()
        modules = []
        modules.append(('bn_relu_conv', BN_ReLU_Conv(in_channel, in_channel, filter_size=(1, 1), stride=(1, 1), pad=(0, 0))))
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules

    def weight_initialization(self):
        self.bn_relu_conv.weight_initialization()

    def __call__(self, x, train=False):
        return F.average_pooling_2d(self.bn_relu_conv(x, train=train), (2, 2), (2, 2), 0)

    def count_parameters(self):
        return self.bn_relu_conv.count_parameters()


class DenseBlock(nutszebra_chainer.Model):

    def __init__(self, in_channel, block_size=32, growth_rate=12):
        super(DenseBlock, self).__init__()
        modules = []
        for i in six.moves.range(1, block_size + 1):
            modules.append(('bn_relu_conv{}'.format(i), BN_ReLU_Conv(in_channel, growth_rate)))
            in_channel = in_channel + growth_rate
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules

    def weight_initialization(self):
        for i in six.moves.range(1, len(self.modules) + 1):
            self['bn_relu_conv{}'.format(i)].weight_initialization()

    def __call__(self, x, train=False):
        for i in six.moves.range(1, len(self.modules) + 1):
            h = self['bn_relu_conv{}'.format(i)](x, train=train)
            x = F.concat([x, h], axis=1)
        return x

    def count_parameters(self):
        count = 0
        for i in six.moves.range(1, len(self.modules) + 1):
            count = count + self['bn_relu_conv{}'.format(i)].count_parameters()
        return count


class DenselyConnectedCNN(nutszebra_chainer.Model):

    def __init__(self, category_num, block_num=3, block_size=32, growth_rate=12):
        super(DenselyConnectedCNN, self).__init__()
        # conv
        modules = [('conv1', L.Convolution2D(3, 16, 3, 1, 1))]
        # dense block
        # if block_num=3, block_size=32, growth_rate=12:
        #    block1's input channel: 16
        #    block1's output channel: 16 + 12 * 32 = 400
        #    block2's input channel: 400
        #    block2's output channel: 400 + 12 * 32 = 784
        #    block3's input channel: 784
        #    block3's output channel: 784 + 12 * 32 = 1168
        #    trans1's input channel: 400
        #    trans1's output channel: 400
        #    trans2's input channel: 784
        #    trans2's output channel: 784
        in_channel = 16
        for i in six.moves.range(1, block_num + 1):
            modules.append(('dense{}'.format(i), DenseBlock(in_channel, block_size, growth_rate)))
            in_channel = in_channel + growth_rate * block_size
            # if block_num=3, then trans1 and trans2 are used
            if i <= block_num - 1:
                modules.append(('trans{}'.format(i), Transition_Layer(in_channel, in_channel)))
        modules.append(('bn1', L.BatchNormalization(in_channel)))
        modules.append(('fc1', L.Linear(in_channel, category_num)))
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.block_num = block_num
        self.block_size = block_size
        self.growth_rate = growth_rate
        self.name = 'densenet_{}_{}_{}_{}'.format(category_num, block_num, block_size, growth_rate)

    def weight_initialization(self):
        self.conv1.W.data = self.weight_relu_initialization(self.conv1)
        self.conv1.b.data = self.bias_initialization(self.conv1, constant=0)
        self.fc1.W.data = self.weight_relu_initialization(self.fc1,)
        self.fc1.b.data = self.bias_initialization(self.fc1, constant=0)
        for i in six.moves.range(1, self.block_num + 1):
            self['dense{}'.format(i)].weight_initialization()
        for i in six.moves.range(1, self.block_num):
            self['trans{}'.format(i)].weight_initialization()

    def __call__(self, x, train=False):
        h = self.conv1(x)
        for i in six.moves.range(1, self.block_num + 1):
            h = self['dense{}'.format(i)](h, train=train)
            if i <= self.block_num - 1:
                h = self['trans{}'.format(i)](h, train=train)
        h = F.relu(self.bn1(h, test=not train))
        num, categories, y, x = h.data.shape
        h = F.reshape(F.average_pooling_2d(h, (y, x)), (num, categories))
        h = self.fc1(h)
        return h

    def count_parameters(self):
        count = 0
        for i in six.moves.range(1, self.block_num + 1):
            count = count + self['dense{}'.format(i)].count_parameters()
            if i <= self.block_num - 1:
                count = count + self['trans{}'.format(i)].count_parameters()
        return count

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy
