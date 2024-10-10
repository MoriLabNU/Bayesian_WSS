import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import pandas as pd

from torch.utils.data import DataLoader
from medpy import metric
from dataloader.dataloader import AutoLaparo_loader
from networks.Bayesian_UNet import BDL_MC_UNet
from utils.visualization_scribble import *


def calculate_metrics(prediction, label):
    if label.sum() == 0 and prediction.sum() == 0:
        dice = np.nan
        IoU = np.nan
        sensitivity = np.nan
        specificity = np.nan

    else:
        dice = metric.dc(prediction, label)
        IoU = metric.jc(prediction, label)
        sensitivity = metric.sensitivity(prediction, label)
        specificity = metric.specificity(prediction, label)

    return dice, IoU, sensitivity, specificity


def validation(args, model, test_dataloader):
    list_dice = []
    list_IoU = []
    list_sensitivity = []
    list_specificity = []
    for i in range(args.num_classes):
        list_dice.append([])
        list_IoU.append([])
        list_sensitivity.append([])
        list_specificity.append([])

    with torch.no_grad():
        for i, test_batch in enumerate(test_dataloader):
            image_batch, label_batch = test_batch['image'].cuda(), test_batch['mask'].cuda()

            outputs = model(image_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs_soft, dim=1)
            prediction = prediction.squeeze().detach().cpu().numpy()
            label = label_batch.squeeze().detach().cpu().numpy()

            for i in range(args.num_classes):
                dice, IoU, sensitivity, specificity = calculate_metrics(prediction == i, label == i)
                list_dice[i].append(dice)
                list_IoU[i].append(IoU)
                list_sensitivity[i].append(sensitivity)
                list_specificity[i].append(specificity)

    return list_dice, list_IoU, list_sensitivity, list_specificity


def validation_MC_dropuout(args, model, test_dataloader, M_dp):
    list_dice = []
    list_IoU = []
    list_sensitivity = []
    list_specificity = []
    for i in range(args.num_classes):
        list_dice.append([])
        list_IoU.append([])
        list_sensitivity.append([])
        list_specificity.append([])

    with torch.no_grad():
        for i, test_batch in enumerate(test_dataloader):
            image_batch, label_batch = test_batch['image'].cuda(), test_batch['mask'].cuda()

            outputs_list = []

            for i in range(M_dp):
                outputs = model(image_batch)
                outputs_soft_ = torch.softmax(outputs, dim=1)

                outputs_list.append(outputs_soft_)

            outputs_soft = torch.stack(outputs_list, dim=0)
            outputs_soft = torch.mean(outputs_soft, dim=0)

            prediction = torch.argmax(outputs_soft, dim=1)
            prediction = prediction.squeeze().detach().cpu().numpy()
            label = label_batch.squeeze().detach().cpu().numpy()

            for i in range(args.num_classes):
                dice, IoU, sensitivity, specificity = calculate_metrics(prediction == i, label == i)
                list_dice[i].append(dice)
                list_IoU[i].append(IoU)
                list_sensitivity[i].append(sensitivity)
                list_specificity[i].append(specificity)

    return list_dice, list_IoU, list_sensitivity, list_specificity


def validation_MC_dropuout_save(args, model, test_dataloader, M_dp, image_save_path):
    list_dice = []
    list_IoU = []
    list_sensitivity = []
    list_specificity = []
    for i in range(args.num_classes):
        list_dice.append([])
        list_IoU.append([])
        list_sensitivity.append([])
        list_specificity.append([])

    idx = 0

    with torch.no_grad():
        for i, test_batch in enumerate(test_dataloader):
            image_batch, label_batch = test_batch['image'].cuda(), test_batch['mask'].cuda()

            outputs_list = []

            for i in range(M_dp):
                outputs = model(image_batch)
                outputs_soft_ = torch.softmax(outputs, dim=1)

                outputs_list.append(outputs_soft_)

            outputs_soft = torch.stack(outputs_list, dim=0)
            outputs_soft = torch.mean(outputs_soft, dim=0)

            entropy = -torch.sum(outputs_soft * torch.log(outputs_soft + 1e-9),
                                 dim=1)

            prediction = torch.argmax(outputs_soft, dim=1)
            prediction = prediction.squeeze().detach().cpu().numpy()
            label = label_batch.squeeze().detach().cpu().numpy()

            for i in range(args.num_classes):
                dice, IoU, sensitivity, specificity = calculate_metrics(prediction == i, label == i)
                list_dice[i].append(dice)
                list_IoU[i].append(IoU)
                list_sensitivity[i].append(sensitivity)
                list_specificity[i].append(specificity)

            # save image and prediction, uncertainty
            # save the image
            image_to_save = image_batch[0].cpu().detach().numpy() * 255
            image_to_save = np.transpose(image_to_save, (1, 2, 0))
            image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_BGR2RGB)
            cv2.imwrite(image_save_path + '/image_{}.png'.format(idx), image_to_save)

            # sava the label
            label_to_save = class_to_rgb(label)
            label_to_save = np.transpose(label_to_save, (1, 2, 0))

            label_to_save = cv2.cvtColor(label_to_save, cv2.COLOR_BGR2RGB)
            cv2.imwrite(image_save_path + '/label_{}.png'.format(idx), label_to_save)

            # ours
            ours_prediction = class_to_rgb(prediction)
            ours_prediction = np.transpose(ours_prediction, (1, 2, 0))
            ours_prediction = cv2.cvtColor(ours_prediction, cv2.COLOR_BGR2RGB)
            cv2.imwrite(image_save_path + '/prediction{}.png'.format(idx), ours_prediction)

            # save uncertainty
            variance_np = entropy[0].cpu().detach().numpy()

            normalized_variance = (variance_np - variance_np.min()) / (variance_np.max() - variance_np.min())

            jet_map = cv2.applyColorMap((normalized_variance * 255).astype(np.uint8), cv2.COLORMAP_JET)

            cv2.imwrite(image_save_path + 'uncertainty{}.png'.format(idx), jet_map)

            idx += 1

    return list_dice, list_IoU, list_sensitivity, list_specificity


def class_to_rgb(predictions):
    colormap = [
        [0, 0, 128],
        [0, 128, 128],
        [128, 0, 128],
        [255, 192, 203],
        [0, 255, 0],
        [165, 42, 42],
        [255, 165, 0],
        [245, 245, 220],
        [255, 215, 0],
        [255, 127, 80],
    ]

    H, W = predictions.shape
    rgb_images = np.zeros((3, H, W), dtype=np.uint8)

    for i in range(10):
        mask = predictions == i
        rgb_images[0, mask] = colormap[i][0]
        rgb_images[1, mask] = colormap[i][1]
        rgb_images[2, mask] = colormap[i][2]

    return rgb_images


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def inference(args):
    snapshot_path = "./snapshot/model_BDL_dense_crf/crf_{}_recon_{}_kl_{}_ST_{}_run{}". \
        format(args.crf, args.recon, args.kl, args.sample_time, args.run)

    if args.DP > 0:
        filename = snapshot_path + "/log_inference_dropout_time_{}.txt".format(args.dp_time)
    else:
        filename = snapshot_path + "/log_inference.txt"

    logging.basicConfig(filename=filename, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    MC_model = BDL_MC_UNet(in_chns=3, class_num=args.num_classes)
    MC_model = MC_model.cuda()
    MC_model.eval()

    model_path = snapshot_path + '/MC_model.pth'
    MC_model.load_state_dict(torch.load(model_path))
    logging.info('load model from {}'.format(model_path))

    current_file_path = __file__
    work_path = os.path.abspath(os.path.dirname(current_file_path))

    val_data = AutoLaparo_loader(work_path=work_path, transform=None, split='val', label_type='dense')
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)

    logging.info('####################################Validation set:')

    if args.DP > 0:
        logging.info('enable dropout')
        enable_dropout(MC_model)
        M_dp = args.dp_time
        list_dice, list_IoU, list_sensitivity, list_specificity = validation_MC_dropuout(args, MC_model, val_dataloader,
                                                                                         M_dp)

    else:
        list_dice, list_IoU, list_sensitivity, list_specificity = validation(args, MC_model, val_dataloader)

    average_dice = 0
    average_IoU = 0
    average_sensitivity = 0
    average_specificity = 0

    assert len(list_dice) == args.num_classes
    assert len(list_IoU) == args.num_classes
    assert len(list_sensitivity) == args.num_classes
    assert len(list_specificity) == args.num_classes

    for i in range(args.num_classes):
        dice_per_class = np.nanmean(list_dice[i])
        IoU_per_class = np.nanmean(list_IoU[i])
        sensitivity_per_class = np.nanmean(list_sensitivity[i])
        specificity_per_class = np.nanmean(list_specificity[i])

        average_dice += dice_per_class
        average_IoU += IoU_per_class
        average_sensitivity += sensitivity_per_class
        average_specificity += specificity_per_class

        logging.info('dice_test of class %d : %f' % (i, dice_per_class))
        logging.info('IoU_test of class %d : %f' % (i, IoU_per_class))
        logging.info('sensitivity_test of class %d : %f' % (i, sensitivity_per_class))
        logging.info('specificity_test of class %d : %f' % (i, specificity_per_class))

    average_dice = average_dice / args.num_classes
    average_IoU = average_IoU / args.num_classes
    average_sensitivity = average_sensitivity / args.num_classes
    average_specificity = average_specificity / args.num_classes

    logging.info('dice_total : %f' % (average_dice))
    logging.info('IoU_total : %f' % (average_IoU))
    logging.info('sensitivity_total : %f' % (average_sensitivity))
    logging.info('specificity_total : %f' % (average_specificity))

    data = {
        'Class_Metric': [],
        'Result': []
    }

    metrics = ['Dice', 'IoU', 'Sensitivity', 'Specificity']
    metric_lists = [list_dice, list_IoU, list_sensitivity, list_specificity]
    averages = [average_dice, average_IoU, average_sensitivity, average_specificity]

    for i in range(args.num_classes):
        for metric, metric_list in zip(metrics, metric_lists):
            class_metric = f'c{i} {metric}'
            result = np.nanmean(metric_list[i])
            data['Class_Metric'].append(class_metric)
            data['Result'].append(result)

    for metric, avg in zip(metrics, averages):
        class_metric = f'avg {metric}'
        data['Class_Metric'].append(class_metric)
        data['Result'].append(avg)

    df = pd.DataFrame(data)

    if args.DP > 0:
        excel_filename = snapshot_path + '/results_val_dropout_time_{}.xlsx'.format(args.dp_time)
    else:
        excel_filename = snapshot_path + '/results_val.xlsx'

    df.to_excel(excel_filename, index=False)

    logging.info(f'Results written to {excel_filename}')

    current_file_path = __file__
    work_path = os.path.abspath(os.path.dirname(current_file_path))

    test_data = AutoLaparo_loader(work_path=work_path, transform=None, split='test', label_type='dense')
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    logging.info('####################################test set:')

    if args.DP > 0:
        logging.info('enable dropout')
        enable_dropout(MC_model)
        M_dp = args.dp_time

        image_save_path = snapshot_path + '/predictions_dropout_time_{}/'.format(args.dp_time)
        os.makedirs(image_save_path, exist_ok=True)

        list_dice, list_IoU, list_sensitivity, list_specificity = validation_MC_dropuout_save(args, MC_model,
                                                                                              test_dataloader, M_dp,
                                                                                              image_save_path)

    else:
        list_dice, list_IoU, list_sensitivity, list_specificity = validation(args, MC_model, test_dataloader)

    average_dice = 0
    average_IoU = 0
    average_sensitivity = 0
    average_specificity = 0

    assert len(list_dice) == args.num_classes
    assert len(list_IoU) == args.num_classes
    assert len(list_sensitivity) == args.num_classes
    assert len(list_specificity) == args.num_classes

    for i in range(args.num_classes):
        dice_per_class = np.nanmean(list_dice[i])
        IoU_per_class = np.nanmean(list_IoU[i])
        sensitivity_per_class = np.nanmean(list_sensitivity[i])
        specificity_per_class = np.nanmean(list_specificity[i])

        average_dice += dice_per_class
        average_IoU += IoU_per_class
        average_sensitivity += sensitivity_per_class
        average_specificity += specificity_per_class

        logging.info('dice_test of class %d : %f' % (i, dice_per_class))
        logging.info('IoU_test of class %d : %f' % (i, IoU_per_class))
        logging.info('sensitivity_test of class %d : %f' % (i, sensitivity_per_class))
        logging.info('specificity_test of class %d : %f' % (i, specificity_per_class))

    average_dice = average_dice / args.num_classes
    average_IoU = average_IoU / args.num_classes
    average_sensitivity = average_sensitivity / args.num_classes
    average_specificity = average_specificity / args.num_classes

    logging.info('dice_total : %f' % (average_dice))
    logging.info('IoU_total : %f' % (average_IoU))
    logging.info('sensitivity_total : %f' % (average_sensitivity))
    logging.info('specificity_total : %f' % (average_specificity))

    data = {
        'Class_Metric': [],
        'Result': []
    }

    metrics = ['Dice', 'IoU', 'Sensitivity', 'Specificity']
    metric_lists = [list_dice, list_IoU, list_sensitivity, list_specificity]
    averages = [average_dice, average_IoU, average_sensitivity, average_specificity]

    for i in range(args.num_classes):
        for metric, metric_list in zip(metrics, metric_lists):
            class_metric = f'c{i} {metric}'
            result = np.nanmean(metric_list[i])
            data['Class_Metric'].append(class_metric)
            data['Result'].append(result)

    for metric, avg in zip(metrics, averages):
        class_metric = f'avg {metric}'
        data['Class_Metric'].append(class_metric)
        data['Result'].append(avg)

    df = pd.DataFrame(data)

    if args.DP > 0:
        excel_filename = snapshot_path + '/results_test_dropout_time_{}.xlsx'.format(args.dp_time)
    else:
        excel_filename = snapshot_path + '/results_test.xlsx'

    df.to_excel(excel_filename, index=False)

    logging.info(f'Results written to {excel_filename}')


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10,
                        help='output channel of network')
    parser.add_argument('--max_epochs', type=int,
                        default=400, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float, default=1e-4,
                        help='segmentation network learning rate')
    parser.add_argument('--run', type=int, default=1, help='trial number')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--kl', type=float, default=0.001, help='alpha: weight of KL divergence loss')
    parser.add_argument('--recon', type=float, default=0.1, help='beta: weight of reconstruction loss')
    parser.add_argument('--crf', type=float, default=1e-8, help='gamma: weight of crf loss')
    parser.add_argument('--sample_time', type=int, default=3, help='N: sample time')
    parser.add_argument('--dp_time', type=int, default=15, help='T: dropout time')
    parser.add_argument('--DP', type=int, default=1, help='>0 to enable dropout')

    args = parser.parse_args()
    return args


def main():
    args = parser_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    inference(args)

if __name__ == "__main__":
    main()
