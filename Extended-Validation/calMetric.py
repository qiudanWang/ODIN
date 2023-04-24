# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import misc


def tpr95(in_file, out_file):
    #calculate the falsepositive error when tpr is 95%
    T = 1000
    cifar = np.loadtxt(in_file, delimiter=',')
    other = np.loadtxt(out_file, delimiter=',')
    # if name == "CIFAR-10": 
    #     start = 0.1
    #     end = 0.12 
    # if name == "CIFAR-100": 
    #     start = 0.01
    #     end = 0.0104    

    # get min and max in both in and out files
    max1 = np.max(cifar[:,2])
    max2 = np.max(other[:,2])
    min1 = np.min(cifar[:,2])
    min2 = np.min(other[:,2])
    start = min(min1.astype(float),min2.astype(float))
    end = max(max1.astype(float),max2.astype(float))
    print("start:",start,",end:",end)
    gap = (end- start)/100000
    #f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1
    fpr = fpr/total
            
    return fpr

def auroc(in_file, out_file):
    #calculate the AUROC
    T = 1000
    cifar = np.loadtxt(in_file, delimiter=',')
    other = np.loadtxt(out_file, delimiter=',')
    # if name == "CIFAR-10": 
    #     start = 0.1
    #     end = 0.12 
    # if name == "CIFAR-100": 
    #     start = 0.01
    #     end = 0.0104    

    # get min and max in both in and out files
    max1 = np.max(cifar[:,2])
    max2 = np.max(other[:,2])
    min1 = np.min(cifar[:,2])
    min2 = np.min(other[:,2])
    start = min(min1.astype(float),min2.astype(float))
    end = max(max1.astype(float),max2.astype(float))
    print("start:",start,",end:",end)
    gap = (end- start)/100000
    #f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    auroc = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / float(len(X1))
        fpr = np.sum(np.sum(Y1 >= delta)) / float(len(Y1))
        auroc += (-fpr+fprTemp)*tpr
        fprTemp = fpr
    auroc += fpr * tpr
    return auroc

def auprIn(in_file, out_file):
    #calculate the AUPR
    T = 1000
    cifar = np.loadtxt(in_file, delimiter=',')
    other = np.loadtxt(out_file, delimiter=',')
    # if name == "CIFAR-10": 
    #     start = 0.1
    #     end = 0.12 
    # if name == "CIFAR-100": 
    #     start = 0.01
    #     end = 0.0104    

    # get min and max in both in and out files
    max1 = np.max(cifar[:,2])
    max2 = np.max(other[:,2])
    min1 = np.min(cifar[:,2])
    min2 = np.min(other[:,2])
    start = min(min1.astype(float),min2.astype(float))
    end = max(max1.astype(float),max2.astype(float))
    print("start:",start,",end:",end)
    gap = (end- start)/100000
    #f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    aupr = 0.0
    recallTemp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta)) / float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        #precisionVec.append(precision)
        #recallVec.append(recall)
        aupr += (recallTemp-recall)*precision
        recallTemp = recall
    aupr += recall * precision
    return aupr

def auprOut(in_file, out_file):
    #calculate the AUPR
    T = 1000
    cifar = np.loadtxt(in_file, delimiter=',')
    other = np.loadtxt(out_file, delimiter=',')
    # if name == "CIFAR-10": 
    #     start = 0.1
    #     end = 0.12 
    # if name == "CIFAR-100": 
    #     start = 0.01
    #     end = 0.0104    

    # get min and max in both in and out files
    max1 = np.max(cifar[:,2])
    max2 = np.max(other[:,2])
    min1 = np.min(cifar[:,2])
    min2 = np.min(other[:,2])
    start = min(min1.astype(float),min2.astype(float))
    end = max(max1.astype(float),max2.astype(float))
    print("start:",start,",end:",end)
    gap = (end- start)/100000
    #f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    aupr = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / float(len(Y1))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        aupr += (recallTemp-recall)*precision
        recallTemp = recall
    aupr += recall * precision
    return aupr



def detection(in_file, out_file):
    #calculate the minimum detection error
    cifar = np.loadtxt(in_file, delimiter=',')
    other = np.loadtxt(out_file, delimiter=',')
    # if name == "CIFAR-10": 
    #     start = 0.1
    #     end = 0.12 
    # if name == "CIFAR-100": 
    #     start = 0.01
    #     end = 0.0104    

    # get min and max in both in and out files
    max1 = np.max(cifar[:,2])
    max2 = np.max(other[:,2])
    min1 = np.min(cifar[:,2])
    min2 = np.min(other[:,2])
    start = min(min1.astype(float),min2.astype(float))
    end = max(max1.astype(float),max2.astype(float))
    print("start:",start,",end:",end)
    gap = (end- start)/100000
    #f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    error = 1.0
    for delta in np.arange(start, end, gap):
        # print(delta)
        tpr = np.sum(np.sum(X1 < delta)) / float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / float(len(Y1))
        error = np.minimum(error, (tpr+error2)/2.0)
            
    return error




def metric(in_file, out_file):
    # if nn == "densenet10" or nn == "wideresnet10": indis = "CIFAR-10"
    # if nn == "densenet10" or nn == "densenet100": nnStructure = "DenseNet-BC-100"
    # if nn == "pointnet2_cls_ssg": indis, nnStructure = "ModelNet10","PointNet"
    # if data == "Imagenet": dataName = "Tiny-ImageNet (crop)"
    fpr = tpr95(in_file, out_file)
    error = detection(in_file, out_file)
    auroc_val = auroc(in_file, out_file)
    auprin = auprIn(in_file, out_file)
    auprout = auprOut(in_file, out_file)
    # print("{:31}{:>22}".format("Neural network architecture:", nnStructure))
    # print("{:31}{:>22}".format("In-distribution dataset:", indis))
    # print("{:31}{:>22}".format("Out-of-distribution dataset:", dataName))
    # print("")
    result = "{:20}{:13.1f}%\n".format("FPR at TPR 95%:", fpr*100) + "{:20}{:13.1f}%\n".format("Detection error:", error*100) + "{:20}{:13.1f}%\n".format("AUROC:", auroc_val*100) + "{:20}{:13.1f}%\n".format("AUPR In:", auprin*100) + "{:20}{:13.1f}%\n".format("AUPR Out:", auprout*100)
    return result








