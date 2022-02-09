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
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from sklearn.metrics import *

def args():
    main_arg_parser = argparse.ArgumentParser(description="parser")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")
    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--gpu", type=int, default=0,
                                  help="assign gpu index")
    train_arg_parser.add_argument("--epochs_finetune", type=int, default=5,
                                  help="train how many epochs for classifier fine-tuning")
    train_arg_parser.add_argument("--train_batchsz", type=int, default=128,
                                  help="batch size for each iteration")
    train_arg_parser.add_argument("--model_name", default='resnet50',
                                  help="resnet18/50/101/152")
    train_arg_parser.add_argument("--train_mode", default='2020only',
                                  help="19+20/2020only")
    train_arg_parser.add_argument("--num_classes", type=int, default=2,
                                  help="number of classes")
    train_arg_parser.add_argument("--lr_finetune", type=float, default=0.0005,
                                  help='learning rate')
    return train_arg_parser.parse_args()


class FocalLoss(nn.Module):
    def __init__(self, gamma=5, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def train_rep(model_name, epochs, train_mode):
    """ Learn Representation """
    netE = torch.load('/path of your saved model', map_location=device)
    netE = netE.to(device)
    netC = classifiers(model_name).to(device)
    optimizer = downstream_optimizer(netC, lr=args().lr_finetune)
    netE.train()
    netC.train()
    loader = loaders(train_mode, train_batchsz=args().train_batchsz)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for iter, (x, y) in enumerate(loader):
            x, y = Variable(x).to(device), Variable(y).to(device)
            features = netC(netE(x))
            loss = loss_fn(features, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        test_acc, AUC = test(netE, netC, loaders('test'), epoch, train_mode, model_name)

def test(netE, netC, test_loader, epoch, train_mode, model_name):
    with torch.no_grad():
        y_pred = np.array([])
        y_true = np.array([])
        for iter, (x, y) in enumerate(test_loader):
            x, y = Variable(x).to(device), Variable(y).to(device)
            predict = netC(netE(x))
            predict_label = torch.max(predict, 1)[1].data.cpu().numpy()
            y_pred = np.append(y_pred, predict_label)
            y_true = np.append(y_true, y.data.cpu().numpy())
        test_acc = accuracy_score(y_true, y_pred)
        score = roc_auc_score(y_true, y_pred)
        targets = ['Benign', 'Malenoma']
        print(classification_report(y_true, y_pred, target_names=targets))
        print('-------------------AUC: {}'.format(score))
        print(confusion_matrix(y_true, y_pred))
    return test_acc, score

if __name__ == "__main__":
    
    device = torch.device('cuda:{}'.format(args().gpu))
    
    print('------------------Using device: {}'.format(device))
    
    print(args())

    train_rep(args().model_name, args().epochs_finetune, args().train_mode)


