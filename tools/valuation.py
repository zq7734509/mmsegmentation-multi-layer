import numpy as np
import os
import glob
from PIL import Image
import cv2 as cv
import os
from sklearn.metrics import confusion_matrix,cohen_kappa_score
from skimage import io
from skimage import measure
from scipy import ndimage
from scipy import misc
from sklearn.metrics import f1_score
from libtiff import TIFF

def evaluateWithBoundary(data_root):
    print('---------------------------------------------------------------------------')
    print('---------------------------------------------------------------------------')
    print('---------------------------------------------------------------------------')
    print('---------------------------------------------------------------------------')
    print('evaluateWithBoundary')
    os.chdir(data_root + 'rgb_img/')
    listName = os.listdir(os.getcwd())
    classNum = 5
    confusion_matrix_total = np.zeros((classNum, classNum), dtype=np.int)
    right_total = 0
    sum_total = 0
    out_txt = open(data_root + 'evaluationWithBoundary.txt', 'w')
    num_total = len(listName)
    num_cur = 1
    for i in listName:
        print('Evaluating process: ' + str(num_cur) + '/' + str(num_total))
        pred_grey_img = Image.open(data_root + "grey_img/" + i[:-8] + "_pred.png")
        pred_grey = np.asarray(pred_grey_img)
        gt_grey_img = Image.open(data_root + "grey_img/" + i[:-8] + "_label.png")
        gt_grey = np.asarray(gt_grey_img)

        out_txt.write(i[:-8] + '_label.png\n')
        out_txt.write('\t\tbuild\tcar\tgrass\timp\ttree\n')
        print(i[:-8] + '_label.png')
        print('\t\tbuild\tcar\tgrass\timp\ttree')
        matrix = confusion_matrix(y_true=np.array(gt_grey).flatten(), y_pred=np.array(pred_grey).flatten())

        other_label_count = 0
        if (gt_grey==0).any() :
            other_label_count = matrix[0, :].sum()
            other_label_count_1 = matrix[:, 0].sum()
            matrix_except_red = matrix[1:, 1:]
        else:
            matrix_except_red = matrix

        matrix_cur = np.zeros((3, classNum), dtype=np.float)
        matrix_normalized = np.zeros((classNum, classNum), dtype=np.float)
        # normalization
        for j in range(classNum):
            matrix_normalized[j, :] = matrix_except_red[j, :] / matrix_except_red[j, :].sum()

        for j in range(classNum):
            precision = np.diag(matrix_except_red)[j] / matrix_except_red[:, j].sum()
            recall = np.diag(matrix_except_red)[j] / matrix_except_red[j, :].sum()
            F1_score = 2 * precision * recall / (precision + recall)
            matrix_cur[0, j] = round(precision, 3)
            matrix_cur[1, j] = round(recall, 3)
            matrix_cur[2, j] = round(F1_score, 3)

        out_txt.write('Precision\t' + str(matrix_cur[0, 0]) + '\t' + str(matrix_cur[0, 1]) + '\t' + str(
            matrix_cur[0, 2]) + '\t' + str(matrix_cur[0, 3]) + '\t' + str(matrix_cur[0, 4]) + '\n')
        out_txt.write('Recall\t\t' + str(matrix_cur[1, 0]) + '\t' + str(matrix_cur[1, 1]) + '\t' + str(
            matrix_cur[1, 2]) + '\t' + str(matrix_cur[1, 3]) + '\t' + str(matrix_cur[1, 4]) + '\n')
        out_txt.write('F1_score\t' + str(matrix_cur[2, 0]) + '\t' + str(matrix_cur[2, 1]) + '\t' + str(
            matrix_cur[2, 2]) + '\t' + str(matrix_cur[2, 3]) + '\t' + str(matrix_cur[2, 4]) + '\n')
        print('Precision\t' + str(matrix_cur[0, 0]) + '\t' + str(matrix_cur[0, 1]) + '\t' + str(
            matrix_cur[0, 2]) + '\t' + str(matrix_cur[0, 3]) + '\t' + str(matrix_cur[0, 4]))
        print('Recall\t\t' + str(matrix_cur[1, 0]) + '\t' + str(matrix_cur[1, 1]) + '\t' + str(
            matrix_cur[1, 2]) + '\t' + str(matrix_cur[1, 3]) + '\t' + str(matrix_cur[1, 4]))
        print('F1_score\t' + str(matrix_cur[2, 0]) + '\t' + str(matrix_cur[2, 1]) + '\t' + str(
            matrix_cur[2, 2]) + '\t' + str(matrix_cur[2, 3]) + '\t' + str(matrix_cur[2, 4]))
        oa = np.diag(matrix_except_red).sum() / (pred_grey.shape[0] * pred_grey.shape[1] - other_label_count)
        aaa = (pred_grey.shape[0] * pred_grey.shape[1] - other_label_count)
        bbb = matrix_except_red.sum()

        out_txt.write('Overall accuracy: ' + str(round(oa, 5)) + '\n')
        out_txt.write('-----------------------------------------------------------------------------\n')
        out_txt.write('\n')
        print('Overall accuracy: ' + str(round(oa, 5)))
        print('-----------------------------------------------------------------------------')
        print('\n')
        confusion_matrix_total += matrix_except_red
        right_total += np.diag(matrix_except_red).sum()
        sum_total += (pred_grey.shape[0] * pred_grey.shape[1] - other_label_count)
        num_cur += 1

    out_txt.write('\n')
    out_txt.write('\n')
    out_txt.write('In Total:\n')
    print('In Total:')
    matrix_total = np.zeros((3, classNum), dtype=np.float)
    for j in range(classNum):
        precision = np.diag(confusion_matrix_total)[j] / confusion_matrix_total[:, j].sum()
        recall = np.diag(confusion_matrix_total)[j] / confusion_matrix_total[j, :].sum()
        F1_score = 2 * precision * recall / (precision + recall)
        matrix_total[0, j] = round(precision, 3)
        matrix_total[1, j] = round(recall, 3)
        matrix_total[2, j] = round(F1_score, 3)
    out_txt.write('Precision\t' + str(matrix_total[0, 0]) + '\t' + str(matrix_total[0, 1]) + '\t' + str(
        matrix_total[0, 2]) + '\t' + str(matrix_total[0, 3]) + '\t' + str(matrix_total[0, 4]) + '\n')
    out_txt.write('Recall\t\t' + str(matrix_total[1, 0]) + '\t' + str(matrix_total[1, 1]) + '\t' + str(
        matrix_total[1, 2]) + '\t' + str(matrix_total[1, 3]) + '\t' + str(matrix_total[1, 4]) + '\n')
    out_txt.write('F1_score\t' + str(matrix_total[2, 0]) + '\t' + str(matrix_total[2, 1]) + '\t' + str(
        matrix_total[2, 2]) + '\t' + str(matrix_total[2, 3]) + '\t' + str(matrix_total[2, 4]) + '\n')
    print('Precision\t' + str(matrix_total[0, 0]) + '\t' + str(matrix_total[0, 1]) + '\t' + str(
        matrix_total[0, 2]) + '\t' + str(matrix_total[0, 3]) + '\t' + str(matrix_total[0, 4]))
    print('Recall\t' + str(matrix_total[1, 0]) + '\t' + str(matrix_total[1, 1]) + '\t' + str(
        matrix_total[1, 2]) + '\t' + str(matrix_total[1, 3]) + '\t' + str(matrix_total[1, 4]))
    print('F1_score\t' + str(matrix_total[2, 0]) + '\t' + str(matrix_total[2, 1]) + '\t' + str(
        matrix_total[2, 2]) + '\t' + str(matrix_total[2, 3]) + '\t' + str(matrix_total[2, 4]))
    oa_total = right_total / sum_total
    out_txt.write('Overall accuracy: ' + str(round(oa_total, 5)) + '\n')
    print('Overall accuracy: ' + str(round(oa_total, 5)))
    out_txt.close()

def evaluateWithNoBoundary(data_root):
    print('---------------------------------------------------------------------------')
    print('---------------------------------------------------------------------------')
    print('---------------------------------------------------------------------------')
    print('---------------------------------------------------------------------------')
    print('evaluateWithNoBoundary')
    os.chdir(data_root + 'rgb_img/')
    listName = os.listdir(os.getcwd())
    classNum = 5
    confusion_matrix_total = np.zeros((classNum, classNum), dtype=np.int)
    right_total = 0
    sum_total = 0
    out_txt = open(data_root + 'evaluationWithNoBoundary.txt', 'w')
    num_total = len(listName)
    num_cur = 1
    for i in listName:
        print('Evaluating process: ' + str(num_cur) + '/' + str(num_total))
        pred_grey_img = Image.open(data_root + "grey_img/" + i[:-8] + "_pred.png")
        pred_grey = np.asarray(pred_grey_img)
        gt_grey_img = Image.open(data_root + "grey_img/" + i[:-8] + "_label_no_boundary.png")
        gt_grey = np.asarray(gt_grey_img)

        out_txt.write(i[:-8] + '_label.png\n')
        out_txt.write('\t\tbuild\tcar\tgrass\timp\ttree\n')
        print(i[:-8] + '_label.png')
        print('\t\tbuild\tcar\tgrass\timp\ttree')
        matrix = confusion_matrix(y_true=np.array(gt_grey).flatten(), y_pred=np.array(pred_grey).flatten())

        matrix_except_red = np.zeros((classNum, classNum), dtype=np.float)
        other_label_count = matrix[0, :].sum()
        other_label_count_1 = matrix[:, 0].sum()
        matrix_except_red = matrix[1:, 1:]

        matrix_cur = np.zeros((3, classNum), dtype=np.float)
        matrix_normalized = np.zeros((classNum, classNum), dtype=np.float)
        # normalization
        for j in range(classNum):
            matrix_normalized[j, :] = matrix_except_red[j, :] / matrix_except_red[j, :].sum()

        for j in range(classNum):
            precision = np.diag(matrix_except_red)[j] / matrix_except_red[:, j].sum()
            recall = np.diag(matrix_except_red)[j] / matrix_except_red[j, :].sum()
            F1_score = 2 * precision * recall / (precision + recall)
            matrix_cur[0, j] = round(precision, 3)
            matrix_cur[1, j] = round(recall, 3)
            matrix_cur[2, j] = round(F1_score, 3)

        out_txt.write('Precision\t' + str(matrix_cur[0, 0]) + '\t' + str(matrix_cur[0, 1]) + '\t' + str(
            matrix_cur[0, 2]) + '\t' + str(matrix_cur[0, 3]) + '\t' + str(matrix_cur[0, 4]) + '\n')
        out_txt.write('Recall\t\t' + str(matrix_cur[1, 0]) + '\t' + str(matrix_cur[1, 1]) + '\t' + str(
            matrix_cur[1, 2]) + '\t' + str(matrix_cur[1, 3]) + '\t' + str(matrix_cur[1, 4]) + '\n')
        out_txt.write('F1_score\t' + str(matrix_cur[2, 0]) + '\t' + str(matrix_cur[2, 1]) + '\t' + str(
            matrix_cur[2, 2]) + '\t' + str(matrix_cur[2, 3]) + '\t' + str(matrix_cur[2, 4]) + '\n')
        print('Precision\t' + str(matrix_cur[0, 0]) + '\t' + str(matrix_cur[0, 1]) + '\t' + str(
            matrix_cur[0, 2]) + '\t' + str(matrix_cur[0, 3]) + '\t' + str(matrix_cur[0, 4]))
        print('Recall\t\t' + str(matrix_cur[1, 0]) + '\t' + str(matrix_cur[1, 1]) + '\t' + str(
            matrix_cur[1, 2]) + '\t' + str(matrix_cur[1, 3]) + '\t' + str(matrix_cur[1, 4]))
        print('F1_score\t' + str(matrix_cur[2, 0]) + '\t' + str(matrix_cur[2, 1]) + '\t' + str(
            matrix_cur[2, 2]) + '\t' + str(matrix_cur[2, 3]) + '\t' + str(matrix_cur[2, 4]))
        oa = np.diag(matrix_except_red).sum() / (pred_grey.shape[0] * pred_grey.shape[1] - other_label_count)

        out_txt.write('Overall accuracy: ' + str(round(oa, 5)) + '\n')
        out_txt.write('-----------------------------------------------------------------------------\n')
        out_txt.write('\n')
        print('Overall accuracy: ' + str(round(oa, 5)))
        print('-----------------------------------------------------------------------------')
        print('\n')
        confusion_matrix_total += matrix_except_red
        right_total += np.diag(matrix_except_red).sum()
        sum_total += (pred_grey.shape[0] * pred_grey.shape[1] - other_label_count)
        num_cur += 1

    out_txt.write('\n')
    out_txt.write('\n')
    out_txt.write('In Total:\n')
    print('In Total:')
    matrix_total = np.zeros((3, classNum), dtype=np.float)
    for j in range(classNum):
        precision = np.diag(confusion_matrix_total)[j] / confusion_matrix_total[:, j].sum()
        recall = np.diag(confusion_matrix_total)[j] / confusion_matrix_total[j, :].sum()
        F1_score = 2 * precision * recall / (precision + recall)
        matrix_total[0, j] = round(precision, 3)
        matrix_total[1, j] = round(recall, 3)
        matrix_total[2, j] = round(F1_score, 3)
    out_txt.write('Precision\t' + str(matrix_total[0, 0]) + '\t' + str(matrix_total[0, 1]) + '\t' + str(
        matrix_total[0, 2]) + '\t' + str(matrix_total[0, 3]) + '\t' + str(matrix_total[0, 4]) + '\n')
    out_txt.write('Recall\t\t' + str(matrix_total[1, 0]) + '\t' + str(matrix_total[1, 1]) + '\t' + str(
        matrix_total[1, 2]) + '\t' + str(matrix_total[1, 3]) + '\t' + str(matrix_total[1, 4]) + '\n')
    out_txt.write('F1_score\t' + str(matrix_total[2, 0]) + '\t' + str(matrix_total[2, 1]) + '\t' + str(
        matrix_total[2, 2]) + '\t' + str(matrix_total[2, 3]) + '\t' + str(matrix_total[2, 4]) + '\n')
    print('Precision\t' + str(matrix_total[0, 0]) + '\t' + str(matrix_total[0, 1]) + '\t' + str(
        matrix_total[0, 2]) + '\t' + str(matrix_total[0, 3]) + '\t' + str(matrix_total[0, 4]))
    print('Recall\t' + str(matrix_total[1, 0]) + '\t' + str(matrix_total[1, 1]) + '\t' + str(
        matrix_total[1, 2]) + '\t' + str(matrix_total[1, 3]) + '\t' + str(matrix_total[1, 4]))
    print('F1_score\t' + str(matrix_total[2, 0]) + '\t' + str(matrix_total[2, 1]) + '\t' + str(
        matrix_total[2, 2]) + '\t' + str(matrix_total[2, 3]) + '\t' + str(matrix_total[2, 4]))
    oa_total = right_total / sum_total
    out_txt.write('Overall accuracy: ' + str(round(oa_total, 5)) + '\n')
    print('Overall accuracy: ' + str(round(oa_total, 5)))
    out_txt.close()

if __name__ == '__main__':
    data_root = '/media/allen/orange/00第二篇论文实验结果/15.upernet_swin_base_potsdam_normal_boundary_loss_160k/'
    need_transform_label_from_rgb2grey = True
    if need_transform_label_from_rgb2grey:
        os.system(
            'sh /home/allen/Documents/MATLAB/run_transform_RGBLabel2Grey.sh /usr/local/MATLAB/MATLAB_Runtime/v93/ ' + data_root)
    evaluateWithBoundary(data_root)
    evaluateWithNoBoundary(data_root)

    os.system(
        'sh /home/allen/Documents/MATLAB/run_generate_red_green.sh /usr/local/MATLAB/MATLAB_Runtime/v93/ ' + data_root)