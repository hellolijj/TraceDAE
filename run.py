from scipy.sparse import data
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse
import scipy.io
import pandas as pd
from sklearn.metrics import roc_auc_score
from datetime import datetime
import argparse

from models.dominant import Dominant
from utils import load_anomaly_detection_dataset
from models.adae import AnomalyDAE
from models.tdae import TraceDAE


def loss_func(adj, A_hat, attrs, X_hat, alpha):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)


    cost =  alpha * attribute_reconstruction_errors + (1-alpha) * structure_reconstruction_errors

    return cost, structure_cost, attribute_cost

def train(args):
    adj, attrs, label, adj_label = load_anomaly_detection_dataset(args.dataset)

    # print(adj) # 二维数组
    # print(attrs) # 二唯数组
    # print(label) # 一维数组
    # print(adj_label) # 而为数组


    adj = torch.FloatTensor(adj)
    adj_label = torch.FloatTensor(adj_label)
    attrs = torch.FloatTensor(attrs)

    if args.model == "adae":
        model = AnomalyDAE(in_dim = attrs.size(1), num_nodes = attrs.size(0), hidden_dim = args.hidden_dim, dropout = args.dropout)
    else args.model == "tdae":
        model = TraceDAE(in_dim=attrs.size(1), num_nodes=attrs.size(0), hidden_dim=args.hidden_dim, dropout=args.dropout)
   
    
    if args.device == 'cuda':
        device = torch.device(args.device)
        adj = adj.to(device)
        adj_label = adj_label.to(device)
        attrs = attrs.to(device)
        model = model.cuda()
        
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)  # 参数调整

    


    # epoch 表示一次训练
    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        # print(attrs.shape, adj.shape)
        A_hat, X_hat = model(attrs, adj)
        # print(A_hat.shape, X_hat.shape)
        # print(adj_label.shape, attrs.shape)
        loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, attrs, X_hat, args.alpha) #损失函数
        l = torch.mean(loss)
        l.backward()
        optimizer.step()        
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "struct_loss=", "{:.5f}".format(struct_loss.item()),"feat_loss=", "{:.5f}".format(feat_loss.item()))

        if epoch%10 == 0 or epoch == args.epoch - 1:
            model.eval()
            A_hat, X_hat = model(attrs, adj)
            loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, attrs, X_hat, args.alpha)
            score = loss.detach().cpu().numpy()
            print("Epoch:", '%04d' % (epoch), 'Auc', roc_auc_score(label, score))  # 实际目标值，决策函数得分


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='BlogCatalog', help='dataset name: Flickr/ACM/BlogCatalog')
    parser.add_argument('--hidden_dim', type=int, default=128, help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--epoch', type=int, default=100, help='Training epoch')  # 训练的次数
    parser.add_argument('--model', default="tdae", help='model choose')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.8, help='balance parameter')
    parser.add_argument('--device', default='cpu', type=str, help='cuda/cpu')
    # parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    

    args = parser.parse_args()

    train(args)