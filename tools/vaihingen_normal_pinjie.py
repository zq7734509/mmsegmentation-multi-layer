#coding:utf-8
import os
from PIL import Image
import numpy as np

OriginImgPath = '/media/allen/orange/mm_dataset/vaihingen_all/image/'   # 不需要改动,只是用来获取每张图片的宽和高

#源目录
SrcPath = '/media/allen/orange/00第二篇论文实验结果/7.upernet_swin_base_vaihingen_normal_240k_rgb+ndsm/small_img/'
#输出目录
OutPath = '/media/allen/orange/00第二篇论文实验结果/7.upernet_swin_base_vaihingen_normal_240k_rgb+ndsm/rgb_img/'

width=600
height=600


def splitList(list_all):
    list_unique = []
    list_name_group = []
    for i in range(len(list_all)):
        ss = list_all[i].split('_')
        ssID = ss[2] + '_' + ss[3]
        if ssID not in list_unique:
            list_unique.append(ssID)
            list_name_group.append([])
            list_name_group[-1].append(list_all[i])
        else:
            list_name_group[list_unique.index(ssID)].append(list_all[i])

    for i in range(len(list_name_group)):
        list_name_group[i].sort()

    return list_name_group




def run():
    #切换到源目录，遍历源目录下所有图片
    os.chdir(SrcPath)
    listName = os.listdir(os.getcwd())
    listNameGroup = splitList(listName)

    for img_num in range(len(listNameGroup)):   # 对于n张大图中的每一幅：
        originImageName = listNameGroup[img_num][0][0:-12] + '.png'
        originImg = Image.open(OriginImgPath + originImageName)
        width1, height1 = originImg.size

        mask_whole = np.zeros((height1, width1, 3), dtype=np.uint8)

        rowheight_half = height // 2
        colwidth_half = width // 2
        row_need_padding = False
        col_need_padding = False

        rownum = height1 // rowheight_half
        colnum = width1 // colwidth_half
        if height1 > rownum * rowheight_half:
            row_need_padding = True
        if width1 > colnum * colwidth_half:
            col_need_padding = True

        for r in range(rownum - 1):
            for c in range(colnum - 1):
                if col_need_padding:
                    print('name: %s' % listNameGroup[img_num][r*colnum+c])
                    img = Image.open(SrcPath + listNameGroup[img_num][r*colnum+c])
                    img_array = np.asarray(img)
                    mask_whole[r * height // 2 : r * height // 2 + height,
                    c * width // 2 : c* width // 2 + width, :] = img_array[:, :, :]
                else:
                    print('name: %s' % listNameGroup[img_num][r * (colnum-1) + c])
                    img = Image.open(SrcPath + listNameGroup[img_num][r * (colnum-1) + c])
                    img_array = np.asarray(img)
                    mask_whole[r * height // 2: r * height // 2 + height,
                    c * width // 2: c * width // 2 + width, :] = img_array[:, :, :]
            if col_need_padding:
                print('name: %s' % listNameGroup[img_num][r*colnum+c+1])
                img = Image.open(SrcPath + listNameGroup[img_num][r*colnum+c+1])
                img_array = np.asarray(img)
                mask_whole[r * height // 2 : r * height // 2 + height, width1-width:width1, :] = img_array[:, :, :]
        if row_need_padding:
            for c in range(colnum - 1):
                if col_need_padding:
                    print('name: %s' % listNameGroup[img_num][(r+1)*colnum+c])
                    img = Image.open(SrcPath + listNameGroup[img_num][(r+1)*colnum+c])
                    img_array = np.asarray(img)
                    mask_whole[height1-height:height1,
                    c * width // 2 : c* width // 2 + width, :] = img_array[:, :, :]
                else:
                    print('name: %s' % listNameGroup[img_num][(r+1) * (colnum-1) + c])
                    img = Image.open(SrcPath + listNameGroup[img_num][(r+1) * (colnum-1) + c])
                    img_array = np.asarray(img)
                    mask_whole[height1-height:height1,
                    c * width // 2: c * width // 2 + width, :] = img_array[:, :, :]
            if col_need_padding:
                print('name: %s' % listNameGroup[img_num][(r+1)*colnum+c+1])
                img = Image.open(SrcPath + listNameGroup[img_num][(r+1)*colnum+c+1])
                img_array = np.asarray(img)
                mask_whole[height1-height:height1, width1-width:width1, :] = img_array[:, :, :]
        mask_whole_img = Image.fromarray(mask_whole)
        mask_whole_img.save(OutPath + listNameGroup[img_num][0][0:-8] + '.png', 'png')





if __name__ == '__main__':
    run()