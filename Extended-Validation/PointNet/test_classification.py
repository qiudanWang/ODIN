"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
import sys
import importlib
from torch.autograd import Variable
import calMetric

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--T', type=int, default=1, help='Tempature')
    parser.add_argument('--noiseMagnitude', type=float, default=0.0, help='noiseMagnitude')
    return parser.parse_args()


def test(model, criterion, loader, file, num_class=40, vote_num=1, T=1, noiseMagnitude=0.0):
    print(T, noiseMagnitude)
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))
    g1 = open(file, 'w')
    g2 = open("./preds/pred_" + str(T) + "_" + str(noiseMagnitude) + "_.txt", "a+")
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()
            classifier = classifier.cuda()
            criterion = criterion.cuda()

        # requires_grad = True
        points = points.transpose(2, 1)
        points.requires_grad_()
        vote_pool = torch.zeros(target.size()[0], num_class)
        if not args.use_cpu:
            vote_pool = vote_pool.cuda()
        # original method
        for _ in range(vote_num):
            norm = None
            B, _, _ = points.shape
            l1_xyz, l1_points = classifier.sa1(points, norm)
            l2_xyz, l2_points = classifier.sa2(l1_xyz, l1_points)
            l3_xyz, l3_points = classifier.sa3(l2_xyz, l2_points)
            x = l3_points.view(B, 1024)
            x = classifier.drop1(F.relu(classifier.bn1(classifier.fc1(x))))
            x = classifier.drop2(F.relu(classifier.bn2(classifier.fc2(x))))
            x = classifier.fc3(x)
            pred = F.log_softmax(x / T, -1)
            vote_pool += pred

        pred = vote_pool / vote_num
        # get gradients
        labels = torch.argmax(pred, dim=1)
        loss = criterion(pred, labels.long(), l3_points)
        loss.backward()
        # add interuption
        gradient =  torch.ge(points.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        tempPoints = torch.add(points.data,  -noiseMagnitude, gradient)
        # add temperature T
        vote_pool = torch.zeros(target.size()[0], num_class)
        if not args.use_cpu:
            vote_pool = vote_pool.cuda()
        for _ in range(vote_num):
            norm = None
            B, _, _ = tempPoints.shape
            l1_xyz, l1_points = classifier.sa1(tempPoints, norm)
            l2_xyz, l2_points = classifier.sa2(l1_xyz, l1_points)
            l3_xyz, l3_points = classifier.sa3(l2_xyz, l2_points)
            x = l3_points.view(B, 1024)
            x = classifier.drop1(F.relu(classifier.bn1(classifier.fc1(x))))
            x = classifier.drop2(F.relu(classifier.bn2(classifier.fc2(x))))
            x = classifier.fc3(x)
            pred = F.log_softmax(x / T, -1)
            vote_pool += pred
        pred = vote_pool / vote_num
        g2.write(str(pred) + "\n")
        pred_choice = pred.data.max(1)[1]
        # print result
        nnOutputs = np.exp(pred.data.cpu().numpy())
        [rows, cols] = nnOutputs.shape
        for idx in range(rows):
            out = str(T) + ", " + str(noiseMagnitude) + ", " + str(np.max(nnOutputs[idx])) + "\n"
            g1.write(out)

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    g1.close()
    g2.close()
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    # in distribution
    shape_names_file = "modelnet10_shape_names.txt"
    test_file = "modelnet10_val_test.txt"
    in_soft_file = "./softmax_scores/confidence_Our_In_" + str(args.T) + "_" + str(args.noiseMagnitude) + "_v.txt"
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False, shape_names_file=shape_names_file, test_file=test_file)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(checkpoint['model_state_dict'])

    instance_acc, class_acc = test(classifier.eval(), model.get_loss(), testDataLoader, T=args.T, noiseMagnitude=args.noiseMagnitude, vote_num=args.num_votes, num_class=num_class, file=in_soft_file)
    log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))

    # out distribution
    test_file = "modelnet10_val_out_test.txt"
    out_soft_file = "./softmax_scores/confidence_Our_Out_" + str(args.T) + "_" + str(args.noiseMagnitude) + "_v.txt"
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False, test_file=test_file)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(checkpoint['model_state_dict'])

    instance_acc, class_acc = test(classifier.eval(), model.get_loss(), testDataLoader, T=args.T, noiseMagnitude=args.noiseMagnitude, vote_num=args.num_votes, num_class=num_class, file=out_soft_file)
    log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))

    met_file = "./metrics/met_" + str(args.T) + "_" + str(args.noiseMagnitude) + "_v.txt"
    met_out = calMetric.metric(in_soft_file, out_soft_file)
    with open(met_file, "w") as f:
        f.write(met_out)

if __name__ == '__main__':
    args = parse_args()
    main(args)
