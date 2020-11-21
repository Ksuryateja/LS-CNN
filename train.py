from math import ceil
import os
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from model import LSCNN
from utils import init_log, Visualizer
from datasets.webface import CASIAWebFace
from torch.optim import lr_scheduler
import torch.optim as optim
import time
import numpy as np
import torchvision.transforms as transforms
import argparse


def train(args):
    # gpu init
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log init
    save_dir = args.save_dir
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    logging = init_log(save_dir)
    _print = logging.info

    # dataset loader
    transform = transforms.Compose([
                transforms.RandomCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
                ])
    # validation dataset
    dataset = CASIAWebFace(args.train_data_info, transform = transform)
    #train_set, val_set = torch.utils.data.random_split(dataset, [int(ceil(0.8 * len(dataset))), len(dataset) - int(ceil(0.8 * len(dataset)))])
    train_set, val_set = torch.utils.data.random_split(dataset, [1, len(dataset) - 1])
    val_set, _ = torch.utils.data.random_split(val_set, [int(ceil(0.01 * len(val_set))), len(val_set) - int(ceil(0.01 * len(val_set)))])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)
    
    net = LSCNN(num_classes= 10559, growth_rate = 48)
    
    if args.resume:
        print('resume the model parameters from: ', args.net_path, args.margin_path)
        net.load_state_dict(torch.load(args.net_path)['net_state_dict'])

    # define optimizers for different layer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.Adam(net.parameters(), lr = 0.001, weight_decay=5e-4)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[12, 16], gamma=0.1)

    if multi_gpus:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)


    best_val_acc = 0.0
    best_val_iters = 0
    total_iters = 0
    #vis = Visualizer()
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
                #vis.plot_curves({'softmax loss': total_loss.item()}, iters=total_iters, title='train loss', xlabel='iters', ylabel='train loss')
                #vis.plot_curves({'train accuracy': correct / total}, iters=total_iters, title='train accuracy', xlabel='iters', ylabel='train accuracy')

                _print("Iters: {:0>6d}/[{:0>2d}], loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}".format(total_iters, epoch, total_loss.item(), correct/total, time_cur, exp_lr_scheduler.get_lr()[0]))

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
                    val_loss = []
                    val_acc = []
                    net.eval()
                    for data in val_loader:
                        img, label = data[0].to(device), data[1].to(device)

                        output = net(img)
                        val_loss.append(criterion(output, label).view(1))
                        _, predict = torch.max(output.data, 1)
                        total = label.size(0)
                        correct = torch.from_numpy(np.array(predict.cpu()) == np.array(label.data.cpu())).sum()
                        val_acc.append(correct.view(1) / total)
                
                val_acc = torch.cat(val_acc, dim=0)
                val_loss = torch.cat(val_loss, dim = 0)
                _print(f'val Ave Accuracy: {torch.mean(val_acc).item() * 100:.4f}, val Ave Loss: {torch.mean(val_loss).item():.4f}')
                if best_val_acc <= torch.mean(val_acc) * 100:
                    best_val_acc = torch.mean(val_acc) * 100
                    best_val_iters = total_iters

                _print(f'Current Best Accuracy: val: {best_val_acc:.4f} in iters: {best_val_iters}')

                #vis.plot_curves({'val': np.mean(val_acc)}, iters=total_iters, title='validation accuracy', xlabel='iters', ylabel='validation accuracy')
                net.train()

    _print('Finally Best Accuracy: val: {:.4f} in iters: {}'.format(best_val_acc, best_val_iters))
    print('finishing training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--train_data_info', type=str, default='/content/data/img_info.csv', help='train image info csv')

    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--total_epoch', type=int, default=18, help='total epochs')

    parser.add_argument('--save_freq', type=int, default=3000, help='save frequency')
    parser.add_argument('--test_freq', type=int, default=1, help='test frequency')
    parser.add_argument('--resume', type=int, default=False, help='resume model')
    parser.add_argument('--net_path', type=str, default='', help='resume model')
    parser.add_argument('--save_dir', type=str, default='./model', help='model save dir')
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='model prefix')

    args = parser.parse_args()

    train(args)
