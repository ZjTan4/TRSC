import torch
import numpy as np
import os
import torch.nn.functional as F
from torchmetrics.classification import ConfusionMatrix

# Adapted from OFFSEG by Viswanath et al. 
def testval(test_dataset, testloader, model, device,
            sv_dir='', sv_pred=False, num_classes=6):
    model.eval()
    compute_confusion_matrix = ConfusionMatrix('multiclass', num_classes=num_classes).to(device)
    confusion_matrix = np.zeros((num_classes, num_classes))
    with torch.no_grad():
        for index, batch in enumerate(testloader):
            image, label, name = batch
            image = image.to(device)
            label = label.to(device)
            size = label.size()
            # pred = multi_scale_inference(
            #     test_dataset, 
            #     model,
            #     image,
            #     scales=[1],
            # )
            pred = model(image)

            # if len(border_padding) > 0:
            #     border_padding = border_padding[0]
            #     pred = pred[:, :, 0:pred.size(2) - border_padding[0], 0:pred.size(3) - border_padding[1]]

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=True
                )

            # confusion_matrix += get_confusion_matrix(
            #     label,
            #     pred,
            #     size,
            #     num_classes)
            pred = pred.permute(0, 2, 3, 1)
            # pred = pred.cpu().numpy().transpose(0, 2, 3, 1)
            pred = torch.argmax(pred, dim=3)
            # pred = np.asarray(np.argmax(pred, axis=3), dtype=np.uint8)
            label = label.squeeze()
 
            # print(pred.unique())
            # print(pred.shape)
            # print(pred[0])
            # print(label.unique())
            # print(label.shape)
            # print(label[0]) 

            confusion_matrix += compute_confusion_matrix(torch.Tensor(pred), torch.Tensor(label)).cpu().numpy()

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            # if index % 100 == 0:
            #     # logging.info('processing: %d images' % index)
            #     pos = confusion_matrix.sum(1)
            #     true = confusion_matrix.sum(0)
            #     tp = np.diag(confusion_matrix)
            #     IoU_array = (tp / np.maximum(1.0, pos + true - tp))
            #     mean_IoU = IoU_array.mean()
            #     # logging.info('mIoU: %.4f' % (mean_IoU))

    return confusion_matrix

def get_IoU(confusion_matrix):
    pos = confusion_matrix.sum(1)
    true = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + true - tp))
    mean_IoU = IoU_array.mean()
    return mean_IoU, IoU_array

def get_Acc(confusion_matrix):
    pos = confusion_matrix.sum(1)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    return mean_acc, pixel_acc

def get_rate(confusion_matrix):
    pos = confusion_matrix.sum(1)
    true = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    fp = pos - tp
    tn = true - tp
    fn = confusion_matrix.sum() - tp - fp - tn
    return tp, fp, tn, fn

def get_Fmeasure(confusion_matrix):
    tp, fp, tn, fn = get_rate(confusion_matrix)
    fm = (2 * tp) / (2 * tp + fp + fn)
    return fm.mean(), fm

def get_confusion_matrix(label, pred, size, num_class,):
# ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.squeeze().cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)
    print(seg_gt.shape)
    print(seg_pred.shape)
    # ignore_index = seg_gt != ignore
    # seg_gt = seg_gt[ignore_index]
    # seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    print(index.shape)
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred] = label_count[cur_index]
    return confusion_matrix