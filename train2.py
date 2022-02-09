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
    train_arg_parser.add_argument("--epochs_contrastive", type=int, default=20,
                                  help="train how many iterations")
    train_arg_parser.add_argument("--train_batchsz", type=int, default=128,
                                  help="batch size for each domain in meta-train, total 64")
    train_arg_parser.add_argument("--head_length", type=int, default=128,
                                  help="batch size for meta test, default is 32")
    train_arg_parser.add_argument("--model_name", default='resnet50',
                                  help="resnet18/50/101/152")
    train_arg_parser.add_argument("--train_mode", default='2020only',
                                  help="resnet18/50/101/152")
    train_arg_parser.add_argument("--num_classes", type=int, default=2,
                                  help="number of classes")
    train_arg_parser.add_argument("--lr_contrastive", type=float, default=0.0005,
                                  help='learning rate of the model')
    return train_arg_parser.parse_args()

device = torch.device('cuda:{}'.format(args().gpu))

print('------------------Using device: {}'.format(device))

print(args())

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

def train_rep(model_name, epochs, n_epoch, train_mode):
    """ Learn Representation """
    netE = torch.load('./modify_model/ISIC_{}_{}_netE_{}.pth'.format(train_mode, model_name, n_epoch),
                      map_location=device)
    netE = netE.to(device)
    netC = classifiers(model_name).to(device)
    optimizer = downstream_optimizer(netC, lr=args().lr_contrastive)
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
        test_acc, AUC = test(netE, netC, loaders('test'), epoch, train_mode, n_epoch, model_name)
        #torch.save(netE,'./modify_model/ISIC_{}_{}_netE_{}.pth'.format(mode, model_name, n_epoch))

def test(netE, netC, test_loader, epoch, train_mode, n_epoch, model_name):
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
        with open('./logs/log_CE_{}_{}_{}.txt'.format(train_mode,model_name,n_epoch), 'a')as file:
            file.write('##################Epoch: {}'.format(epoch+1))
            file.write('\n')
            file.write(classification_report(y_true, y_pred, target_names=targets))
            file.write('\n')
            file.write('-----------------AUC socre: {}'.format(score))
            file.write('\n')
    return test_acc, score

stop_epochs = [100]
models = ['resnet101']
modes = ['19+20']
#for train_mode in modes:
#    for model in models:
#        for numb in stop_epochs:
#            print('------------------processing:{}, {} on {}'.format(train_mode, numb, model))
#            train_rep(model_name=model, epochs=20, n_epoch=numb, train_mode=train_mode)

train_mode = '2020only'
model = 'resnet18'
print('------------------processing:{}, {} on {}'.format(train_mode, 100, model))
train_rep(model_name=model, epochs=20, n_epoch=100, train_mode=train_mode)




