import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from utils import rep_optimizer, downstream_optimizer, class_optimizer
from Models import rep_nets, classifiers
from get_data import loaders
from losses import SupConLoss
import argparse
from datetime import datetime

def args():
    main_arg_parser = argparse.ArgumentParser(description="parser")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")
    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--gpu", type=int, default=0,
                                  help="assign gpu index")
    train_arg_parser.add_argument("--epochs_contrastive", type=int, default=100,
                                  help="train how many epochs")
    train_arg_parser.add_argument("--train_batchsz", type=int, default=128,
                                  help="batch size for each iteration")
    train_arg_parser.add_argument("--head_length", type=int, default=128,
                                  help="the output dimension of mapping module")
    train_arg_parser.add_argument("--model_name", default='resnet50',
                                  help="resnet18/50/101/152")
    train_arg_parser.add_argument("--train_mode", default='2020only',
                                  help="19+20/2020only")
    train_arg_parser.add_argument("--num_classes", type=int, default=2,
                                  help="number of classes")
    train_arg_parser.add_argument("--lr_contrastive", type=float, default=0.003,
                                  help='learning rate of the model')
    return train_arg_parser.parse_args()


def train_rep(model_name, epochs):
    """ Learn Representation """
    netE, rep_head = rep_nets(model_name, len_reps=args().head_length)
    netE, rep_head = netE.to(device), rep_head.to(device)
    rep_optim = rep_optimizer(netE, rep_head, lr=args().lr_contrastive)
    netE.train()
    rep_head.train()
    loader = loaders(args().train_mode, train_batchsz=args().train_batchsz)
    loss_fn = SupConLoss(args().gpu)
    scheduler = torch.optim.lr_scheduler.StepLR(rep_optim, step_size=int(epochs/3), gamma=0.1)
    for epoch in range(epochs):
        for iter, (original, pretext, y) in enumerate(loader):
            original, pretext, y = Variable(original).to(device), \
                                   Variable(pretext).to(device), \
                                   Variable(y).to(device)
            images = torch.cat([original, pretext], 0)
            batchsz = y.shape[0]
            features = rep_head(netE(images))
            feature1, feature2 = torch.split(features, [batchsz, batchsz], dim=0)
            features = torch.cat([feature1.unsqueeze(1), feature2.unsqueeze(1)], dim=1)
            loss = loss_fn(features, y)
            rep_optim.zero_grad()
            loss.backward()
            rep_optim.step()
        scheduler.step()
        now = datetime.now()
        print('Time: {}:{}, Epoch: {:03d}/{}, Contrastive loss: {:.8f}'.format(now.hour, now.minute, epoch+1, epochs, loss.data.cpu().numpy()))
    torch.save(netE.state_dict(), './path_to_save_your_model.pth')


if __name__ == "__main__":
    
    device = torch.device('cuda:{}'.format(args().gpu))
    
    print('------------------Using device: {}'.format(device))
    
    print(args())

    train_rep(model_name=args().model_name, epochs=args().epochs_contrastive)
