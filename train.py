from math import ceil
import os
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from model import LSCNN
from utils import init_log
from datasets.webface import CASIAWebFace
from datasets.lfw import LFW
from eval_lfw import evaluation_10_fold, getFeatureFromTorch
from torch.optim import lr_scheduler
import torch.optim as optim
import time
import numpy as np
import torchvision.transforms as transforms
import argparse
from torch.utils.tensorboard import SummaryWriter

def train(args):
    writer = SummaryWriter()

    # gpu init
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log init
    save_dir = args.save_dir
    if os.path.exists(save_dir):
        if not args.resume:
            raise NameError('model dir exists!')
    else:
        os.makedirs(save_dir)
    logging = init_log(save_dir)
    _print = logging.info

    # dataset loader
    train_transform = transforms.Compose([
                transforms.RandomCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
                ])
    # validation dataset
    train_set = CASIAWebFace(args.train_data_info, transform = train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=False)
    
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(128),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    lfw_dataset = LFW('./data/lfw_funneled', './data/lfw_funneled/pairs.txt', transform = test_transform)
    lfw_dataloader = torch.utils.data.DataLoader(lfw_dataset, batch_size= 128, shuffle=False, num_workers=2, drop_last=False)
    
    net = LSCNN(num_classes= 10559, growth_rate = 48)
    
    if args.resume:
        print('resume the model parameters from: ', args.net_path)
        net.load_state_dict(torch.load(args.net_path)['net_state_dict'])

    # define optimizers for different layer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD(net.parameters(), lr = 0.1, weight_decay = 1e-4, momentum = 0.9, nesterov = True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[10, 20, 30], gamma=0.1)
    
    if multi_gpus:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)

    best_test_acc = 0.0
    best_test_iters = 0
    total_iters = 0

    for epoch in range(1, args.total_epoch + 1):
        exp_lr_scheduler.step()
        # train model
        _print('Train Epoch: {}/{} ...'.format(epoch, args.total_epoch))
        net.train()

        since = time.time()
        for data in train_loader:
            img, label = data[0].to(device), data[1].to(device)
            optimizer_ft.zero_grad()

            
            output = net(img)
            total_loss = criterion(output, label)
            total_loss.backward()
            optimizer_ft.step()

            total_iters += 1
            # print train information
            if total_iters % 100 == 0:
                # current training accuracy
                _, predict = torch.max(output.data, 1)
                total = label.size(0)
                correct = (np.array(predict.cpu()) == np.array(label.data.cpu())).sum()
                time_cur = (time.time() - since) / 100
                since = time.time()
                
                writer.add_scalar("Accuracy/train", correct / total , total_iters)
                writer.add_scalar("Loss/train", total_loss, total_iters)

                _print("Iters: {:0>6d}/[{:0>2d}], loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}".format(total_iters, epoch, total_loss.item(), correct/total, time_cur, exp_lr_scheduler.get_last_lr()))

            # save model
            if total_iters % args.save_freq == 0:
                msg = 'Saving checkpoint: {}'.format(total_iters)
                _print(msg)
                if multi_gpus:
                    net_state_dict = net.module.state_dict()
                else:
                    net_state_dict = net.state_dict()

                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': net_state_dict},
                    os.path.join(save_dir, 'Iter_%06d_net.ckpt' % total_iters))

            # test accuracy
            if total_iters % args.test_freq == 0:
                with torch.no_grad():
                    net.eval()
                    getFeatureFromTorch('./result/cur_epoch_lfw_result.mat', net, device, lfw_dataset, lfw_dataloader)
                    lfw_accuracy = evaluation_10_fold('./result/cur_epoch_lfw_result.mat')
                    lfw_accuracy = np.mean(lfw_accuracy) * 100             
                writer.add_scalar("Accuracy/test", lfw_accuracy, total_iters)
                _print(f'LFW Ave Accuracy: {lfw_accuracy.item():.4f}')
                if best_test_acc <= lfw_accuracy.item() :
                    best_test_acc = lfw_accuracy.item() 
                    best_test_iters = total_iters

                _print(f'Current Best Accuracy: test: {best_test_acc:.4f} in iters: {best_test_iters}')
                net.train()

    _print('Finally Best Accuracy: val: {:.4f} in iters: {}'.format(best_test_acc, best_test_iters))
    print('finishing training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--train_data_info', type=str, default='./data/img_info.csv', help='train image info csv')

    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--total_epoch', type=int, default=25, help='total epochs')

    parser.add_argument('--save_freq', type=int, default=3500, help='save frequency')
    parser.add_argument('--test_freq', type=int, default=3500, help='test frequency')
    parser.add_argument('--resume', type=bool, default=False, help='resume model')
    parser.add_argument('--net_path', type=str, default='', help='resume model')
    parser.add_argument('--save_dir', type=str, default='./model', help='model save dir')
    parser.add_argument('--gpus', type=str, default='0', help='model prefix')

    args = parser.parse_args()

    train(args)