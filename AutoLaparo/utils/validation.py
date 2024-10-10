# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from medpy import metric

import cv2

from medpy import metric

def general_dice(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0


    for instrument_id in set(y_true.flatten()):
        result += [metric.dc(y_pred == instrument_id, y_true == instrument_id)]

    return np.mean(result)


def general_jaccard(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [jaccard(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection

    if union == 0:
        return np.nan

    else:
        return intersection / union


def validation(model, test_dataloader):
    model.eval()
    list = []
    for i in range(10):
        list.append([])

    with torch.no_grad():
        for i, test_batch in enumerate(test_dataloader):
            image_batch, label_batch = test_batch['image'].cuda(), test_batch['mask'].cuda()

            outputs = model(image_batch)
            outputs_soft = torch.softmax(outputs, dim = 1)
            prediction = torch.argmax(outputs_soft, dim = 1)
            prediction = prediction.squeeze().detach().cpu().numpy()
            label = label_batch.squeeze().detach().cpu().numpy()

            for i in range(10):
                list[i].append(jaccard(prediction == i, label == i))

    model.train()

    return list


def validation_MC_dropuout(model, test_dataloader, M_dp):
    list = []
    for i in range(10):
        list.append([])

    with torch.no_grad():
        for i, test_batch in enumerate(test_dataloader):
            image_batch, label_batch = test_batch['image'].cuda(), test_batch['mask'].cuda()

            outputs_list = []

            for i in range(M_dp):

                outputs = model(image_batch)
                outputs_soft_ = torch.softmax(outputs, dim = 1)

                outputs_list.append(outputs_soft_)

            outputs_soft = torch.stack(outputs_list, dim = 0)
            outputs_soft = torch.mean(outputs_soft, dim = 0)

            prediction = torch.argmax(outputs_soft, dim = 1)
            prediction = prediction.squeeze().detach().cpu().numpy()
            label = label_batch.squeeze().detach().cpu().numpy()

            for i in range(10):
                list[i].append(jaccard(prediction == i, label == i))

    return list


def validation_L_model(model, test_dataloader):
    model.eval()
    list = []
    for i in range(10):
        list.append([])
    with torch.no_grad():
        for i, test_batch in enumerate(test_dataloader):
            image_batch, label_batch = test_batch['image'].cuda(), test_batch['mask'].cuda()

            _, _, _, outputs = model(image_batch, M = 1)
            outputs_soft = torch.softmax(outputs[0], dim = 1)
            prediction = torch.argmax(outputs_soft, dim = 1)
            prediction = prediction.squeeze().detach().cpu().numpy()
            label = label_batch.squeeze().detach().cpu().numpy()

            for i in range(10):
                list[i].append(jaccard(prediction == i, label == i))

    model.train()

    return list

def validation_db(model, test_dataloader):
    model.eval()
    list = []
    for i in range(10):
        list.append([])
    with torch.no_grad():
        for i, test_batch in enumerate(test_dataloader):
            image_batch, label_batch = test_batch['image'].cuda(), test_batch['mask'].cuda()

            outputs, _ = model(image_batch)
            outputs_soft = torch.softmax(outputs, dim = 1)
            prediction = torch.argmax(outputs_soft, dim = 1)
            prediction = prediction.squeeze().detach().cpu().numpy()
            label = label_batch.squeeze().detach().cpu().numpy()

            for i in range(10):
                list[i].append(jaccard(prediction == i, label == i))

    model.train()

    return list


def denormalize(image, mean, std):
    image = image.permute(1, 2, 0) # Changes the dimensions from (C, H, W) to (H, W, C)
    image = image * std + mean # denormalize
    image = image.clamp(0, 1) # clamp the values to the range [0,1] if necessary
    return image


def validation_and_save(model1,model2,model3, model4,model5,model6, test_dataloader, snapshot_path):

    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    model6.eval()

    with torch.no_grad():
        for i, test_batch in enumerate(test_dataloader):
            image_batch, label_batch, file_name = test_batch['image'].cuda(), test_batch['mask'].cuda(), test_batch['name']


            outputs1 = model1(image_batch)
            outputs_soft1 = torch.softmax(outputs1, dim = 1)
            prediction1 = torch.argmax(outputs_soft1, dim = 1)
            prediction1 = prediction1.squeeze().detach().cpu().numpy()

            outputs2 = model2(image_batch)
            outputs_soft2 = torch.softmax(outputs2, dim = 1)
            prediction2 = torch.argmax(outputs_soft2, dim = 1)
            prediction2 = prediction2.squeeze().detach().cpu().numpy()

            outputs3 = model3(image_batch)
            outputs_soft3 = torch.softmax(outputs3, dim = 1)
            prediction3 = torch.argmax(outputs_soft3, dim = 1)
            prediction3 = prediction3.squeeze().detach().cpu().numpy()

            outputs4 = model4(image_batch)
            outputs_soft4 = torch.softmax(outputs4, dim = 1)
            prediction4 = torch.argmax(outputs_soft4, dim = 1)
            prediction4 = prediction4.squeeze().detach().cpu().numpy()

            outputs5 = model5(image_batch)
            outputs_soft5 = torch.softmax(outputs5, dim = 1)
            prediction5 = torch.argmax(outputs_soft5, dim = 1)
            prediction5 = prediction5.squeeze().detach().cpu().numpy()

            outputs6 = model6(image_batch)
            outputs_soft6 = torch.softmax(outputs6, dim = 1)
            prediction6 = torch.argmax(outputs_soft6, dim = 1)
            prediction6 = prediction6.squeeze().detach().cpu().numpy()

            label = label_batch.squeeze().detach().cpu().numpy()

            score1 = general_jaccard(y_true = label, y_pred = prediction1)
            score2 = general_jaccard(y_true = label, y_pred = prediction2)
            score3 = general_jaccard(y_true = label, y_pred = prediction3)
            score4 = general_jaccard(y_true = label, y_pred = prediction4)
            score5 = general_jaccard(y_true = label, y_pred = prediction5)
            score6 = general_jaccard(y_true = label, y_pred = prediction6)

            if score6 > score5 and score5 > score4 and score5 > score3 and score5 > score2 and score5 > score1:
                # Save original image, label and prediction
                # Assuming original image is in 'image' key and has been normalized
                image = cv2.imread(file_name[0])
                height, width, _ = image.shape
                image = cv2.resize(image, (width // 4, height // 4))

                print('the {}-th image is from {}'.format(i, file_name[0]))


                # Convert label and prediction to [0, 255]
                label = (label_batch.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
                prediction1 = (prediction1 * 255).astype(np.uint8)
                prediction2 = (prediction2 * 255).astype(np.uint8)
                prediction3 = (prediction3 * 255).astype(np.uint8)
                prediction4 = (prediction4 * 255).astype(np.uint8)
                prediction5 = (prediction5 * 255).astype(np.uint8)
                prediction6 = (prediction6 * 255).astype(np.uint8)


                # Save the original image
                # Save the label
                cv2.imwrite(f'{snapshot_path}/image_{i}.png', image)
                cv2.imwrite(f'{snapshot_path}/label_{i}.png', label)
                # Save the prediction
                cv2.imwrite(f'{snapshot_path}/prediction_{i}_1.png', prediction1)
                cv2.imwrite(f'{snapshot_path}/prediction_{i}_2.png', prediction2)
                cv2.imwrite(f'{snapshot_path}/prediction_{i}_3.png', prediction3)
                cv2.imwrite(f'{snapshot_path}/prediction_{i}_4.png', prediction4)
                cv2.imwrite(f'{snapshot_path}/prediction_{i}_5.png', prediction5)
                cv2.imwrite(f'{snapshot_path}/prediction_{i}_6.png', prediction6)

    return