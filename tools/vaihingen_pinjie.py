#coding:utf-8
import os
from PIL import Image
import numpy as np

srcPath = '/media/allen/orange/mm_dataset/eval_result/vaihingen/resnest50-dnl-boundary-vaihingen-600/'
outPath = '/media/allen/orange/mm_dataset/eval_result/vaihingen/6000x6000/'
originImgPath = '/media/allen/orange/ISPRS dataset/Vaihingen/allen/image/val_png/'

width=600
height=600

def splitList(list_all):
    list_unique = []
    list_origin_img_name = []
    list_name_group = []
    for i in range(len(list_all)):
        ss = list_all[i].split('_')
        if ss[3] not in list_unique:
            list_unique.append(ss[3])
            list_origin_img_name.append(originImgPath + ss[0] + '_' + ss[1] + '_' + ss[2] + '_' + ss[3] + '.png')
            list_name_group.append([])
            list_name_group[-1].append(list_all[i])
        else:
            list_name_group[list_unique.index(ss[3])].append(list_all[i])

    for i in range(len(list_name_group)):
        list_name_group[i].sort()

    return list_name_group, list_origin_img_name

def run():
    #切换到源目录，遍历源目录下所有图片
    os.chdir(srcPath)
    listName = os.listdir(os.getcwd())
    listNameGroup, listOriginImgName = splitList(listName)
    for img_num in range(len(listNameGroup)):
        origin_img = Image.open(listOriginImgName[img_num])
        origin_img_array = np.asarray(origin_img)
        ori_h,ori_w,_ = origin_img_array.shape
        mask_whole = np.zeros((ori_h, ori_w, 3), dtype=np.uint8)
        height_num = ori_h // height + 1
        width_num = ori_w // width + 1
        for i in range(height_num-1):
            for j in range(width_num-1):
                print('name: %s' % listNameGroup[img_num][i*width_num+j])
                img = Image.open(srcPath + listNameGroup[img_num][i*width_num+j])
                img_array = np.asarray(img)
                mask_whole[i*height:i*height+height,j*width:j*width+width,:] = img_array[:,:,:]
            print('name: %s' % listNameGroup[img_num][i * width_num + width_num - 1])
            img = Image.open(srcPath + listNameGroup[img_num][i * width_num + width_num - 1])
            img_array = np.asarray(img)
            mask_whole[i*height:i*height+height, ori_w-width:ori_w, :] = img_array[:, :, :]

        for k in range(width_num - 1):
            print('name: %s' % listNameGroup[img_num][(height_num-1) * width_num + k])
            img = Image.open(srcPath + listNameGroup[img_num][(height_num-1) * width_num + k])
            img_array = np.asarray(img)
            mask_whole[ori_h-height:ori_h, k * width:k * width + width, :] = img_array[:, :, :]
        print('name: %s' % listNameGroup[img_num][(height_num-1) * width_num + width_num - 1])
        img = Image.open(srcPath + listNameGroup[img_num][(height_num-1) * width_num + width_num - 1])
        img_array = np.asarray(img)
        mask_whole[ori_h-height:ori_h, ori_w - width:ori_w, :] = img_array[:, :, :]
        mask_whole_img = Image.fromarray(mask_whole)
        mask_whole_img.save(outPath + listNameGroup[img_num][0][0:-8], 'png')



if __name__ == '__main__':
    run()