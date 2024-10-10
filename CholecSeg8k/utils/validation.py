import torch
import numpy as np

def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection

    if union == 0:
        return np.nan

    else:
        return intersection / union


def validation(model, test_dataloader):
    model.eval()
    metric_list = []
    for i in range(13):
        metric_list.append([])

    with torch.no_grad():
        for i, test_batch in enumerate(test_dataloader):
            image_batch, label_batch = test_batch['image'].cuda(), test_batch['mask'].cuda()

            outputs = model(image_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs_soft, dim=1)
            prediction = prediction.squeeze().detach().cpu().numpy()
            label = label_batch.squeeze().detach().cpu().numpy()

            for i in range(13):
                metric_list[i].append(jaccard(prediction == i, label == i))

    model.train()

    return metric_list


def validation_L_model(model, test_dataloader):
    model.eval()
    metric_list = []
    for i in range(13):
        metric_list.append([])
    with torch.no_grad():
        for i, test_batch in enumerate(test_dataloader):
            image_batch, label_batch = test_batch['image'].cuda(), test_batch['mask'].cuda()

            _, _, _, outputs = model(image_batch, M=1)
            outputs_soft = torch.softmax(outputs[0], dim=1)
            prediction = torch.argmax(outputs_soft, dim=1)
            prediction = prediction.squeeze().detach().cpu().numpy()
            label = label_batch.squeeze().detach().cpu().numpy()

            for i in range(13):
                metric_list[i].append(jaccard(prediction == i, label == i))

    model.train()

    return metric_list
