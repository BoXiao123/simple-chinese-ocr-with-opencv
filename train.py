#!/usr/bin/env python
# -*- coding: utf8 -*-

from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
import mxnet as mx
from mxnet.gluon.data import vision
from time import time


head = '%(asctime)-5s %(message)s'
import logging
logging.basicConfig(level=logging.DEBUG, format=head)

#数据增广
def transform_train(data,label):
    im = image.imresize(data.astype('float32') / 255, 32, 32)
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), resize=0,
                                    rand_crop=False, rand_resize=False, rand_mirror=True,
                                    mean=None, std=None,
                                    brightness=0, contrast=0,
                                    saturation=0, hue=0,
                                    pca_noise=0, rand_gray=0, inter_method=2)
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1))
    return (im, nd.array([label]).asscalar().astype('float32'))

def transform_test(data, label):
    im = image.imresize(data.astype('float32') / 255, 32, 32)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))
#定义残差模块
from mxnet.gluon import nn
from mxnet import nd
class Residual(nn.HybridBlock):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1,
                                  strides=strides)
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm()
            if not same_shape:
                self.conv3 = nn.Conv2D(channels, kernel_size=1,
                                      strides=strides)

    def hybrid_forward(self, F, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out + x)

#定义残差网络
class ResNet(nn.HybridBlock):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            net = self.net = nn.HybridSequential()

            net.add(nn.Conv2D(channels=32, kernel_size=3, strides=1,
                              padding=1))
            net.add(nn.BatchNorm())
            net.add(nn.Activation(activation='relu'))

            for _ in range(3):
                net.add(Residual(channels=32))

            net.add(Residual(channels=64, same_shape=False))
            for _ in range(2):
                net.add(Residual(channels=64))

            net.add(Residual(channels=128, same_shape=False))
            for _ in range(2):
                net.add(Residual(channels=128))

            net.add(nn.GlobalAvgPool2D())
            net.add(nn.Flatten())
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out
#初始化网络
def get_net(ctx):
    num_outputs=30
    net=ResNet(num_outputs)
    net.initialize(ctx=ctx,init=init.Xavier())
    return net

def _get_batch(batch, ctx):
    """return data and label on ctx"""
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.label[0]
    else:
        data, label = batch
    return (gluon.utils.split_and_load(data, ctx),
            gluon.utils.split_and_load(label, ctx),
            data.shape[0])

def evaluate_accuracy(data_iterator, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0])
    n = 0.
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()
    for batch in data_iterator:
        data, label, batch_size = _get_batch(batch, ctx)
        for X, y in zip(data, label):
            acc += nd.sum(net(X).argmax(axis=1)==y).copyto(mx.cpu())
            n += y.size
        acc.wait_to_read() # don't push too many operators into backend
    return acc.asscalar() / n

#定义训练
def train(train_data, test_data, net, loss, trainer, ctx, num_epochs, print_batches=None):
    """Train a network"""
    print("Start training on ", ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        train_loss, train_acc, n, m = 0.0, 0.0, 0.0, 0.0
        if isinstance(train_data, mx.io.MXDataIter):
            train_data.reset()
        start = time()
        for i, batch in enumerate(train_data):
            data, label, batch_size = _get_batch(batch, ctx)
            losses = []
            with autograd.record():
                outputs = [net(X) for X in data]
                losses = [loss(yhat, y) for yhat, y in zip(outputs, label)]
            for l in losses:
                l.backward()
            train_acc += sum([(yhat.argmax(axis=1)==y).sum().asscalar()
                              for yhat, y in zip(outputs, label)])
            train_loss += sum([l.sum().asscalar() for l in losses])
            trainer.step(batch_size)
            n += batch_size
            m += sum([y.size for y in label])
            if print_batches and (i+1) % print_batches == 0:
                print("Batch %d. Loss: %f, Train acc %f" % (
                    n, train_loss/n, train_acc/m
                ))

        test_acc = evaluate_accuracy(test_data, net, ctx)
        print("Epoch %d. Loss: %.3f, Train acc %.2f, Test acc %.2f, Time %.1f sec" % (
            epoch, train_loss/n, train_acc/m, test_acc, time() - start
        ))

#预测结果
def classify(fname):
    train_ds = vision.ImageFolderDataset('train', flag=1,
                                         transform=transform_train)
    with open(fname, 'rb') as f:
        img = image.imdecode(f.read())
    data=image.imresize(img.astype('float32') / 255, 32, 32)
    data = nd.transpose(data, (2, 0, 1))
    data=data.expand_dims(axis=0)
    net=get_net(mx.cpu(0))
    net.load_params('model.params',mx.cpu(0))
    out = net(data.as_in_context(mx.cpu(0)))
    out = nd.SoftmaxActivation(out)
    pred = int(nd.argmax(out, axis=1).asscalar())
    label = train_ds.synsets
    return label[pred]



if __name__=='__main__':
    batch_size=32
    train_ds = vision.ImageFolderDataset('train', flag=1,
                                         transform=transform_train)
    valid_ds = vision.ImageFolderDataset('valid', flag=1,
                                         transform=transform_test)
    loader = gluon.data.DataLoader
    train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
    valid_data = loader(valid_ds, batch_size, shuffle=True, last_batch='keep')
    ctx=mx.gpu(0)
    epochs=100
    learning_rate=0.01
    wd=0.0001
    net=get_net(ctx)
    net.hybridize()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': wd})
    train(train_data, valid_data, net, loss, trainer, ctx, epochs, print_batches=10)
    net.save_params('model.params')

