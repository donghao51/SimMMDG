import os
import torch
from torch.optim import *
import torchvision
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import numpy as np
import json
import argparse
import csv
# from model import AVENet
# from datasets import GetAudioVideoDataset


class opt():
    def __init__(self):
        opt.data_path = '/scratch/shared/beegfs/hchen/train_data/VGGSound_final/audio/'
        opt.result_path = '/scratch/shared/beegfs/hchen/prediction/audioclassification/vggsound/resnet18/'
        opt.summaries = '/scratch/shared/beegfs/hchen/epoch/audioclassification_f/resnet18_vlad/model.pth.tar'
        opt.pool = "avgpool"
        opt.csv_path = './data/'
        opt.test = 'test.csv'
        opt.batch_size = 32
        opt.n_classes = 309
        opt.model_depth = 18
        opt.resnet_shortcut = 'B'

def get_arguments():
    opt1 = opt()
    return opt1



# def main():
#     args = get_arguments()
#
#     # create prediction directory if not exists
#     if not os.path.exists(args.result_path):
#         os.mkdir(args.result_path)
#
#     # init network
#     os.environ["CUDA_VISIBLE_DEVICES"]="0"
#     model= AVENet(args)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model.cuda()
#
#     # load pretrained models
#     checkpoint = torch.load(args.summaries)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.to(device)
#     print('load pretrained model.')
#
#     # create dataloader
#     testdataset = GetAudioVideoDataset(args,  mode='test')
#     testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False,num_workers = 16)
#
#     softmax = nn.Softmax(dim=1)
#     print("Loaded dataloader.")
#
#     model.eval()
#     for step, (spec, audio, label, name) in enumerate(testdataloader):
#         print('%d / %d' % (step,len(testdataloader) - 1))
#         spec = Variable(spec).cuda()
#         label = Variable(label).cuda()
#         aud_o = model(spec.unsqueeze(1).float())
#
#         prediction = softmax(aud_o)
#
#         for i, item in enumerate(name):
#             np.save(args.result_path + '/%s.npy' % item,prediction[i].cpu().data.numpy())
#
#             # print example scores
#             # print('%s, label : %s, prediction score : %.3f' % (
#             #     name[i][:-4], testdataset.classes[label[i]], prediction[i].cpu().data.numpy()[label[i]]))
#
#
#
# if __name__ == "__main__":
#     main()
#
