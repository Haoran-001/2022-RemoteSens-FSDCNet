import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os, shutil, json
import argparse
import time

from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from tools.utils import record_times
from models.FSDCNet import MVCNN, SVCNN

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="FSDCNet")
# Batch size is set to 128 for single view, 16 for 6 views and 8 for 12 views.
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=16)
parser.add_argument("-num_models", type=int, help="number of models per class", default=1000)
parser.add_argument("-lr", type=float, help="learning rate", default=1e-4)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=1e-4)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_false', default = True)
parser.add_argument("--pretrained_model", type=str, help="your pretrained model on ImageNet")
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="dy_resnet50")
parser.add_argument("-num_views", type=int, help="number of views", default=6)

parser.add_argument("-train_path", type=str, default='data_train_path')
parser.add_argument("-val_path", type=str, default="data_test_path")
parser.set_defaults(train=False)

def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)


if __name__ == '__main__':
    args = parser.parse_args()
    pretraining = not args.no_pretraining

    log_dir = args.name
    create_folder(args.name)
    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()

    print('###################Stage 1####################')
    # create FSDCNet_stage_1 folder, for single view train and test
    log_dir = args.name+'_stage_1'
    create_folder(log_dir)

    # 导入预训练模型，从预训练模型的参数开始优化训练，采用模型为vgg-11和resnet-50
    cnet = SVCNN(args.name, nclasses=40, pretraining=pretraining, pretrained_model=args.pretrained_model, cnn_name=args.cnn_name) # ModelNet40
    # cnet = SVCNN(args.name, nclasses = 26, pretraining = pretraining, pretrained_model=args.pretrained_model, cnn_name = args.cnn_name) # Sydney urban object

    cnet_ = torch.nn.DataParallel(cnet, device_ids=[0, 1])
    n_models_train = args.num_models * args.num_views
    optimizer = optim.Adam(cnet_.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = 5)

    print('Loading traning set and val set!')
    train_dataset = SingleImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=10, drop_last = False)
    val_dataset = SingleImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=10, drop_last = False)

    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))

    trainer = ModelNetTrainer(cnet_, train_loader, val_loader, optimizer, cos_scheduler, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_views=1)
    tic1 = time.process_time()
    trainer.train(n_epochs=30)
    toc1 = time.process_time()
    print('The training time of first stage: %d m' % ((toc1-tic1)/60))

    # create FSDCNet_stage_2 folder, for 6 and 12 views train and test
    print('###################Stage 2####################')
    log_dir = args.name+'_stage_2'
    create_folder(log_dir)

    # cnet_2 and cnet, there are same network model.
    cnet_2 = MVCNN(args.name, cnet, pool_mode='gap', nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views) # ModelNet40
    # cnet_2 = MVCNN(args.name, cnet, pool_mode = 'gap', nclasses = 26, cnn_name = args.cnn_name, num_views = args.num_views)
    del cnet
    del cnet_
    cnet_2 = torch.nn.DataParallel(cnet_2, device_ids=[0, 1])
    optimizer = optim.Adam(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=10, drop_last = False) # shuffle needs to be false! it's done within the trainer
    val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_views=args.num_views)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=10, drop_last = False)

    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))

    trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, cos_scheduler, nn.CrossEntropyLoss(), 'mvcnn', log_dir = log_dir, num_views=args.num_views)
    tic2 = time.process_time()
    trainer.train(n_epochs=30) 
    toc2 = time.process_time()
    print('The training time of second stage:%d m' % ((toc2-tic2)/60))
    record_times((toc1-tic1)/60, (toc2-tic2)/60, 'records.txt')