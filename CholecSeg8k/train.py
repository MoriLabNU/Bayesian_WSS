import argparse
import logging
import os
import sys
import random
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Compose,
)
from dataloader.dataloader import CholecSeg8k_loader
from networks.Bayesian_UNet import BDL_L_UNet, BDL_MC_UNet
from utils.validation import validation, validation_L_model
from utils.visualization_scribble import *
from utils.crfloss.pytorch_deeplab_v3_plus.DenseCRFLoss import DenseCRFLoss


def train(args):
    snapshot_path = "./snapshot/model_BDL_dense_crf/crf_{}_recon_{}_kl_{}_ST_{}_fold{}". \
        format(args.crf, args.recon, args.kl, args.sample_time, args.fold)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    writer = SummaryWriter(snapshot_path + '/log')

    L_model = BDL_L_UNet(in_chns=3, class_num=args.num_classes)
    L_model = L_model.cuda()
    L_model.train()

    MC_model = BDL_MC_UNet(in_chns=3, class_num=args.num_classes)
    MC_model = MC_model.cuda()
    MC_model.train()

    L_optimizer = optim.Adam(L_model.parameters(), lr=args.base_lr, weight_decay=0.0001)
    MC_optimizer = optim.Adam(MC_model.parameters(), lr=args.base_lr, weight_decay=0.0001)

    train_transform = Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5)
    ], p=1)

    current_file_path = __file__
    work_path = os.path.abspath(os.path.dirname(current_file_path))

    train_data = CholecSeg8k_loader(work_path= work_path, transform=train_transform, split='train', label_type='scribble', fold=args.fold)
    test_data = CholecSeg8k_loader(work_path= work_path, transform=None, split='val', fold=args.fold, label_type='dense')

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    base_lr = args.base_lr
    iter_num = 0

    sample_time = args.sample_time

    losslayer = DenseCRFLoss(weight=args.crf, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)

    logging.info('############start modeling p(x,y|z)')
    for round_idx in tqdm(range(1, args.max_epochs // 2 + 1), ncols=70):

        batch_loss = []
        for i, train_batch in enumerate(train_dataloader):
            image_batch, label_batch = train_batch['image'].cuda(), train_batch['mask'].cuda()

            mu, log_var, x_gen, y = L_model(image_batch, sample_time=sample_time)

            loss_kl = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

            x_gen = torch.mean(x_gen, dim=0)
            y = torch.softmax(y, dim=2)
            y = torch.mean(y, dim=0)

            y_log = torch.log(y + 1e-12)

            loss_recon = F.mse_loss(x_gen, image_batch)

            loss_pce = F.nll_loss(y_log, label_batch, ignore_index=13)

            outputs_soft = y

            B, _, H, W = outputs_soft.shape

            loss_crf = losslayer(image_batch.clone().cpu() * 255.0, outputs_soft, torch.ones(B, 1, H, W))

            loss_crf = loss_crf.cuda()

            loss = loss_pce + loss_crf + args.recon * loss_recon + args.kl * loss_kl

            L_optimizer.zero_grad()
            loss.backward()
            L_optimizer.step()

            iter_num = iter_num + 1

            if iter_num % 100 == 0:
                logging.info(
                    'iteration %d : lr : %f, loss : %f, loss_recon : %f, loss_kl : %f, loss_pce : %f, loss_crf : %f'
                    % (iter_num, base_lr, loss.item(), loss_recon.item(), loss_kl.item(), loss_pce.item(),
                       loss_crf.item()))

            if iter_num % 1000 == 0:

                img_grid = make_grid(image_batch[0:4], nrow=8)
                writer.add_image('images', img_grid, iter_num)

                x_recon = make_grid(x_gen[0:4], nrow=8)
                writer.add_image('x_recon', x_recon, iter_num)

                label_to_save = label_batch[0:4].cpu().numpy()
                label_to_save = torch.from_numpy(dilation_color_mapping(label_to_save)).permute(0, 3, 1, 2)

                label_grid = make_grid(label_to_save, nrow=8)
                writer.add_image('labels', label_grid, iter_num)

                pred_to_save = torch.argmax(torch.softmax(y, dim=1), dim=1)[0:4]
                pred_to_save = class_to_rgb(pred_to_save)
                pred_grid = make_grid(pred_to_save,
                                      nrow=8)
                writer.add_image('pred', pred_grid, iter_num)

            batch_loss.append(loss.item())

        logging.info('p(x,y|z)_round_idx %d loss : %f' % (round_idx, np.array(batch_loss).mean()))
        writer.add_scalar('p(x,y|z)_loss/', np.array(batch_loss).mean(), round_idx)

        if round_idx % 50 == 0:

            torch.save(L_model.state_dict(), snapshot_path + '/L_model_{}.pth'.format(round_idx))

            val_L_metric_list = validation_L_model(L_model, test_dataloader)

            val_L_metric_average = 0
            for i in range(len(val_L_metric_list)):
                per_class = np.nanmean(val_L_metric_list[i])
                val_L_metric_average += per_class
                logging.info('p(x,y|z)_round_idx %d test_avg of class %d : %f'
                             % (round_idx, i, per_class))

            val_L_metric_average = val_L_metric_average / len(val_L_metric_list)

            logging.info('round_idx %d test_avg : %f' % (round_idx, val_L_metric_average))
            writer.add_scalar('p(x,y|z)_test_avg/', val_L_metric_average, round_idx)
            L_model.train()

    torch.save(L_model.state_dict(), snapshot_path + '/L_model.pth')

    L_model.eval()
    logging.info("L_model is in eval mode")

    logging.info('############start modeling p(w|x,y)')

    iter_num = 0

    for round_idx in tqdm(range(1, args.max_epochs // 2 + 1), ncols=70):

        batch_loss = []

        for i, train_batch in enumerate(train_dataloader):
            image_batch, label_batch = train_batch['image'].cuda(), train_batch['mask'].cuda()

            with torch.no_grad():
                _, _, _, y = L_model(image_batch, sample_time=sample_time)

            y = torch.softmax(y, dim=2)
            y = torch.mean(y, dim=0)
            y = torch.argmax(y, dim=1)

            # merge scribble and generated pseudo label
            mask = torch.zeros_like(y)
            mask[label_batch == 13] = 1

            merged_label = label_batch * (1 - mask) + y * mask

            MC_output = MC_model(image_batch)

            assert merged_label.requires_grad == False

            loss_sup1 = F.cross_entropy(MC_output, merged_label)

            loss = loss_sup1

            MC_optimizer.zero_grad()
            loss.backward()
            MC_optimizer.step()

            iter_num = iter_num + 1

            if iter_num % 100 == 0:
                logging.info(
                    'iteration %d : lr : %f, loss : %f'
                    % (iter_num, base_lr, loss.item()))

            batch_loss.append(loss.item())

        logging.info('p(w|x,y)_round_idx %d loss : %f' % (round_idx, np.array(batch_loss).mean()))
        writer.add_scalar('p(w|x,y)_loss/', np.array(batch_loss).mean(), round_idx)

        if round_idx % 50 == 0:

            torch.save(MC_model.state_dict(), snapshot_path + '/MC_model_{}.pth'.format(round_idx))

            val_MC_metric_list = validation(MC_model, test_dataloader)
            val_MC_metric_average = 0
            for i in range(len(val_MC_metric_list)):
                per_class = np.nanmean(val_MC_metric_list[i])
                val_MC_metric_average += per_class
                logging.info('p(w|x,y)_round_idx %d test_avg of class %d : %f'
                             % (round_idx, i, per_class))

            val_MC_metric_average = val_MC_metric_average / len(val_MC_metric_list)

            logging.info('round_idx %d test_avg : %f' % (round_idx, val_MC_metric_average))
            writer.add_scalar('p(W|,XY)_test_avg/', val_MC_metric_average, round_idx)
            MC_model.train()

    torch.save(MC_model.state_dict(), snapshot_path + '/MC_model.pth')
    writer.close()


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=13,
                        help='output channel of network')
    parser.add_argument('--max_epochs', type=int,
                        default=200, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float, default=1e-4,
                        help='segmentation network learning rate')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--deterministic', type=int, default=0, help='whether use deterministic training')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--fold', type=int, default=1, help='fold to use')
    parser.add_argument('--kl', type=float, default=0.001, help='alpha: weight of KL divergence loss')
    parser.add_argument('--recon', type=float, default=0.1, help='beta: weight of reconstruction loss')
    parser.add_argument('--crf', type=float, default=1e-8, help='gamma: weight of crf loss')
    parser.add_argument('--sample_time', type=int, default=3, help='N: sample time')

    args = parser.parse_args()
    return args


def main():
    args = parser_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()
