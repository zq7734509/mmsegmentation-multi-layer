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
    os.chdir(data_root + 'grey_img/')
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

        pred_grey_img = Image.open(data_root + "grey_img/" + i[:])
        pred_grey = np.asarray(pred_grey_img)
        gt_grey_img = Image.open('/media/allen/orange/mm_dataset/mydataset_Potsdam_600_normal/ann_dir/val/' + i[:-9] + ".png")
        gt_grey = np.asarray(gt_grey_img)

        out_txt.write(i[:-9] + '_label.png\n')
        out_txt.write('\t\tbuild\tcar\tgrass\timp\ttree\n')
        print(i[:-9] + '_label.png')
        print('\t\tbuild\tcar\tgrass\timp\ttree')
        matrix = confusion_matrix(y_true=np.array(gt_grey).flatten(), y_pred=np.array(pred_grey).flatten())

        other_label_count = 0
        if (gt_grey==0).any() :
            other_label_count = matrix[0, :].sum()
            matrix_except_red = matrix[1:, 1:]
        else:
            matrix_except_red = matrix
        oa = np.diag(matrix_except_red).sum() / (pred_grey.shape[0] * pred_grey.shape[1] - other_label_count)
        out_txt.write('Overall accuracy: ' + str(round(oa, 5)) + '\n')
        out_txt.write('-----------------------------------------------------------------------------\n')
        out_txt.write('\n')
        print('Overall accuracy: ' + str(round(oa, 5)))
        print('-----------------------------------------------------------------------------')
        print('\n')
        right_total += np.diag(matrix_except_red).sum()
        sum_total += (pred_grey.shape[0] * pred_grey.shape[1] - other_label_count)
        num_cur += 1

    out_txt.write('\n')
    out_txt.write('\n')
    out_txt.write('In Total:\n')
    print('In Total:')

    oa_total = right_total / sum_total
    out_txt.write('Overall accuracy: ' + str(round(oa_total, 5)) + '\n')
    print('Overall accuracy: ' + str(round(oa_total, 5)))
    out_txt.close()

if __name__ == '__main__':
    data_root = '/home/allen/show/'
    need_transform_label_from_rgb2grey = False
    if need_transform_label_from_rgb2grey:
        os.system(
            'sh /home/allen/Documents/MATLAB/run_transform_RGBLabel2Grey.sh /usr/local/MATLAB/MATLAB_Runtime/v93/ ' + data_root)
    evaluateWithBoundary(data_root)