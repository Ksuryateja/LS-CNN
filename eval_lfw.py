import numpy as np
import scipy.io
import os
import json
import torch.utils.data
from model import  LSCNN
from datasets.lfw import LFW
import torchvision.transforms as transforms
from torch.nn import DataParallel
import argparse

def getAccuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] > threshold)
    n = np.sum(scores[flags == -1] < threshold)
    return 1.0 * (p + n) / len(scores)

def getThreshold(scores, flags, thrNum):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])
    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold

def evaluation_10_fold(feature_path='./result/cur_epoch_result.mat'):
    ACCs = np.zeros(10)
    result = scipy.io.loadmat(feature_path)
    for i in range(10):
        fold = result['fold']
        flags = result['flag']
        featureLs = result['fl']
        featureRs = result['fr']

        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)

        mu = np.mean(np.concatenate((featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0)
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs - mu
        featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

        scores = np.sum(np.multiply(featureLs, featureRs), 1)
        threshold = getThreshold(scores[valFold[0]], flags[valFold[0]], 10000)
        ACCs[i] = getAccuracy(scores[testFold[0]], flags[testFold[0]], threshold)

    return ACCs

def loadModel(data_root, file_list, gpus='0', resume=None):


    # gpu init
    multi_gpus = False
    if len(gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net = LSCNN(10559, growth_rate = 48)
    checkpoint = torch.load(resume, map_location='cpu')
    net.load_state_dict(checkpoint['net_state_dict'])

    if multi_gpus:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(128),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    lfw_dataset = LFW(data_root, file_list, transform=transform)
    lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=128, shuffle=False, num_workers=2, drop_last=False)

    return net.eval(), device, lfw_dataset, lfw_loader

def getFeatureFromTorch(feature_save_dir, net, device, data_set, data_loader):
    featureLs = None
    featureRs = None
    count = 0
    for data in data_loader:
        for i in range(len(data)):
            data[i] = data[i].to(device)
        count += data[0].size(0)
        #print('extracing deep features from the face pair {}...'.format(count))

        activation = {}
        def get_activation(name: str):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        net = net
        
        net.fc.register_forward_hook(get_activation('fc'))
        
        res =[]
        with torch.no_grad():
            for d in data:
                output = net(d)
                res.append(activation['fc'].data.cpu().numpy())
            #res = [net(d).data.cpu().numpy() for d in data]
        featureL = np.concatenate((res[0], res[1]), 1)
        featureR = np.concatenate((res[2], res[3]), 1)
        # print(featureL.shape, featureR.shape)
        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = np.concatenate((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = np.concatenate((featureRs, featureR), 0)
        # print(featureLs.shape, featureRs.shape)

    result = {'fl': featureLs, 'fr': featureRs, 'fold': data_set.folds, 'flag': data_set.flags}
    scipy.io.savemat(feature_save_dir, result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--root', type=str, default='./data/lfw_funneled', help='The path of lfw data')
    parser.add_argument('--file_list', type=str, default='./data/lfw_funneled/pairs.txt', help='The path of lfw pairs.txt')
    parser.add_argument('--resume', type=str, default='./model/Iter_035700_net.ckpt',
                        help='The path pf save model')
    parser.add_argument('--feature_save_path', type=str, default='./result/cur_epoch_lfw_result.mat',
                        help='The path of the extract features save, must be .mat file')
    parser.add_argument('--gpus', type=str, default='0', help='gpu list')
    args = parser.parse_args()

    net, device, lfw_dataset, lfw_loader = loadModel(args.root, args.file_list, args.gpus, args.resume)
    getFeatureFromTorch(args.feature_save_path, net, device, lfw_dataset, lfw_loader)
    ACCs = evaluation_10_fold(args.feature_save_path)
    for i in range(len(ACCs)):
        print('{}    {:.2f}'.format(i+1, ACCs[i] * 100))
    print('--------')
    print('AVE    {:.4f}'.format(np.mean(ACCs) * 100))