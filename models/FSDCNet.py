import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .Model import Model
from torchsummary import summary
import models.dy_resnet as dy_resnet


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class SVCNN(Model):
    def __init__(self, name, nclasses=40, pretraining=False, pretrained_model="", cnn_name='dy_resnet50'):
        super(SVCNN, self).__init__(name)

        # ModelNet40
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        # Sydney urban objects
        # self.classnames = ['4wd', 'bench', 'bicycle', 'biker', 'building', 'bus', 'car', 'cyclist', 'excavator',
        # 'pedestrian', 'pillar', 'pole', 'post', 'scooter', 'ticket_machine', 'traffic_lights',
        # 'traffic_sign', 'trailer', 'trash', 'tree', 'truck', 'trunk', 'umbrella', 'ute', 'van', 'vegetation']

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.use_resnext = cnn_name.startswith('resnext')
        self.use_densenet = cnn_name.startswith('densenet')
        self.use_dy_resnet = cnn_name.startswith('dy_resnet')

        self.mean = Variable(torch.FloatTensor([0.0142, 0.0142, 0.0142]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.0818, 0.0818, 0.0818]), requires_grad=False).cuda()

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, 40)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, 40)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, 40)
            elif self.cnn_name == 'resnet50x':
                self.net = models.resnext50_32x4d(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, 40)
            elif self.cnn_name == 'resnet101x':
                self.net = models.resnext101_32x8d(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, 40)
        elif self.use_resnext:
            if self.cnn_name == 'resnext50':
                self.net = models.resnext50_32x4d(pretrained = self.pretraining)
                self.net.fc = nn.Linear(2048, 40)
        elif self.use_densenet:
            if self.cnn_name =='densenet121':
                self.net = models.densenet121(pretrained=self.pretraining)
                self.net.classifier = nn.Linear(1024, 40)
        elif self.use_dy_resnet:
            if self.cnn_name == 'dy_resnet50':
                self.net = dy_resnet.resnet50(self.pretraining)
                self.net.fc = nn.Linear(2048, self.nclasses)
                if pretraining:
                    state_dict = torch.load(pretrained_model)
                    self.net.load_state_dict(state_dict)

        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretraining).features
                self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier

            self.net_2._modules['6'] = nn.Linear(4096, 40)

    def forward(self, x):
        # print(x.size())
        if self.use_resnet:
            return self.net(x)
        elif self.use_resnext:
            return self.net(x)
        elif self.use_densenet:
            return self.net(x)
        elif self.use_dy_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            # print(y.size())
            return self.net_2(y.view(y.shape[0], -1))


class MVCNN(Model):
    def __init__(self, name, model, pool_mode='max', nclasses=40, cnn_name='dy_resnet50', num_views=6):
        super(MVCNN, self).__init__(name)
        # ModelNet40
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        # Sydney urban objects
        # self.classnames = ['4wd', 'bench', 'bicycle', 'biker', 'building', 'bus', 'car', 'cyclist', 'excavator',
        #                    'pedestrian', 'pillar', 'pole', 'post', 'scooter', 'ticket_machine', 'traffic_lights',
        #                    'traffic_sign', 'trailer', 'trash', 'tree', 'truck', 'trunk', 'umbrella', 'ute', 'van',
        #                    'vegetation']
        
        self.nclasses = nclasses
        self.num_views = num_views

        self.use_resnet = cnn_name.startswith('resnet')
        self.use_densenet = cnn_name.startswith('densenet')
        self.use_dy_resnet = cnn_name.startswith('dy_resnet')
        self.num_conv = 2
        self.pool_mode = pool_mode

        if self.pool_mode not in ['conv', 'attention', 'max', 'joint', 'max+mean', 'gap']:
            raise ValueError("Pool mode is not right! Please input one of 'conv', 'attention', 'add', 'joint' or 'max.")

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
        elif self.use_densenet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.pool = nn.AvgPool2d(kernel_size=7, stride=7, padding=0)
            self.net_2 = nn.Sequential(nn.Linear(1024, 40))
        elif self.use_dy_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2

        if self.pool_mode =='conv':
            self.conv = nn.Conv2d(1, self.num_conv, (1, num_views), padding_mode='valid')
            # self.class_conv = nn.Conv1d(2, 2, (2048), padding_mode='valid')
            self.net_2 = nn.Sequential(nn.Linear(2048*self.num_conv, 40))
        elif self.pool_mode == 'attention':
            # attention
            self.conv1 = nn.Conv2d(1, 1, 1, bias=False)
            self.conv2 = nn.Conv2d(1, 1, 1, bias=False)
            self.conv3 = nn.Conv2d(1, 1, 1, bias=False)
            self.net_2 = nn.Sequential(nn.Linear(2048, 40))
        elif self.pool_mode == 'joint':
            self.jointconv = nn.Conv1d(3, 1, 1, bias=False)
            self.net_2 = nn.Sequential(nn.Linear(2048, self.nclasses))
        elif self.pool_mode == 'max':
            self.net_2 = nn.Sequential(nn.Linear(2048, self.nclasses))
        elif self.pool_mode == 'max+mean':
            self.net_2 = nn.Sequential(nn.Linear(4096, self.nclasses))
        elif self.pool_mode == 'gap':
            self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
            self.SE = nn.Sequential(
                nn.Conv2d(self.num_views, self.num_views, kernel_size = (1, 1)),
                nn.SiLU(),
                nn.Conv2d(self.num_views, self.num_views, kernel_size = (1, 1)),
                nn.Softmax(dim = 1),
            )
            self.net_2 = nn.Sequential(nn.Linear(2048, self.nclasses))

    def forward(self, x):
        y = self.net_1(x)
        # print('y:', y.size())
        # y = self.pool(y)
        # print('y:', y.size())

        if self.pool_mode == 'conv':
            y = y.view((int(x.shape[0] / self.num_views), 1, y.shape[-3], self.num_views))
            # print('the reshape of y:', y.size())
            # ConvPooling
            conv_y = self.conv(y)
            # print('conv_y', conv_y.size())
            y = conv_y.view(y.shape[0], -1)
        elif self.pool_mode == 'attention':
            y = y.view((int(x.shape[0] / self.num_views), 1, y.shape[-3], self.num_views))
            # Attention
            q = self.conv1(y)
            k = self.conv2(y)
            v = self.conv3(y)

            s = torch.matmul(torch.transpose(q, 2, 3), k)
            # print(s.size())
            beta = torch.nn.functional.softmax(s)
            # print(beta.size())
            o = torch.matmul(v, beta)
            # print(o.size())
            gamma = torch.autograd.Variable(torch.FloatTensor([[1.]]), requires_grad=True).cuda()
            y = y + gamma * o
            y = torch.max(y, 3)[0].view(y.shape[0], -1)
            # print(y.size())
        elif self.pool_mode == 'add':
            y = y.view((int(x.shape[0] / self.num_views), self.num_views, y.shape[-3], y.shape[-2], y.shape[-1]))
            y1 = torch.max(y, 1)[0]
            y2 = torch.mean(y, 1)
            y3 = torch.median(y, 1)[0]
            y = (y1 + y2 + y3).view(y.shape[0], -1)
        elif self.pool_mode == 'joint':
            y = y.view((int(x.shape[0] / self.num_views), self.num_views, y.shape[-3], y.shape[-2], y.shape[-1]))
            y1 = torch.max(y, 1)[0]
            y2 = torch.mean(y, 1)
            y3 = torch.median(y, 1)[0]
            y = torch.cat((y1, y2, y3), 2)
            # print('the cat of y', y.size())
            y = self.jointconv(y.view(y.shape[0], y.shape[2], y.shape[1])).view(y.shape[0], -1)
        elif self.pool_mode =='max+mean':
            y = y.view((int(x.shape[0] / self.num_views), self.num_views, y.shape[-3], y.shape[-2], y.shape[-1]))
            y1 = torch.max(y, 1)[0]
            y2 = torch.mean(y, 1)
            y = torch.cat((y1, y2), 2).view(y.shape[0], -1)
        elif self.pool_mode == 'max':
            y = y.view((int(x.shape[0] / self.num_views), self.num_views, y.shape[-3], y.shape[-2], y.shape[-1]))
            y = torch.max(y, 1)[0].view(y.shape[0], -1)

        # Our global attention pool
        elif self.pool_mode == 'gap':
            y1 = y.view(int(x.shape[0] / self.num_views), self.num_views, 1, y.shape[-3])
            weight = self.global_avg(y1)
            weight = self.SE(weight)

            y = (y1 * weight).squeeze(2).sum(1)

        return self.net_2(y)


if __name__ == '__main__':
    cnn_name = 'resnet50'
    name = 'MVCNN'
    cnet = SVCNN(name, nclasses=40, pretraining=True, cnn_name=cnn_name)
    model = MVCNN(name, cnet, nclasses=40, cnn_name=cnn_name, num_views=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    summary(model, input_size=(3, 224, 224))
