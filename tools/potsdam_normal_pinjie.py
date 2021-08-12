#coding:utf-8
import os
from PIL import Image
import numpy as np
#源目录
MyPath = '/media/allen/orange/00第二篇论文实验结果/15.upernet_swin_base_potsdam_normal_boundary_loss_160k/small_img/'
#输出目录
OutPath = '/media/allen/orange/00第二篇论文实验结果/15.upernet_swin_base_potsdam_normal_boundary_loss_160k/rgb_img/'
width=600
height=600
width1=6000
height1=6000
width_half_num=width1 // width * 2 - 1
height_half_num=height1 // height * 2 - 1

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
    os.chdir(MyPath)
    listName = os.listdir(os.getcwd())
    listNameGroup = splitList(listName)
    mask_whole = np.zeros((height1,width1,3),dtype=np.uint8)
    mask_whole1 = np.zeros((height1,width1,3),dtype=np.uint8)

    for img_num in range(len(listNameGroup)):   # 对于14张大图中的每一幅：
        for i in range(width_half_num):
            # 第一行的所有图片
            print('name: %s' % listNameGroup[img_num][i])
            img = Image.open(MyPath + listNameGroup[img_num][i])
            img_array = np.asarray(img)
            mask_whole1[0:height,i*width//2:i*width//2+width,:] = img_array[:,:,:]

            # 最后一行的所有图片
            print('name: %s' % listNameGroup[img_num][(height_half_num - 1) * width_half_num + i])
            img = Image.open(MyPath + listNameGroup[img_num][(height_half_num - 1) * width_half_num + i])
            img_array = np.asarray(img)
            mask_whole1[height1-height:height1, i * width // 2:i * width // 2 + width, :] = img_array[:, :, :]

        for j in range(height_half_num):
            # 第一列的所有图片
            print('name: %s' % listNameGroup[img_num][j * width_half_num])
            img = Image.open(MyPath + listNameGroup[img_num][j * width_half_num])
            img_array = np.asarray(img)
            mask_whole1[j*height//2:j*height//2+height,0:width,:] = img_array[:,:,:]

            # 最后一列的所有图片
            print('name: %s' % listNameGroup[img_num][j * width_half_num + width_half_num - 1])
            img = Image.open(MyPath + listNameGroup[img_num][j * width_half_num + width_half_num - 1])
            img_array = np.asarray(img)
            mask_whole1[j*height//2:j*height//2+height, width1-width:width1, :] = img_array[:, :, :]

        for i in range(width_half_num):
            for j in range(height_half_num):
                print('name: %s' % listNameGroup[img_num][j * width_half_num + i])
                img = Image.open(MyPath + listNameGroup[img_num][j * width_half_num + i])
                img_array = np.asarray(img)
                mask_whole1[j*height//2+height // 4:j*height//2+height // 4 * 3, i*width//2+width // 4:i*width//2+width//4*3, :] = img_array[height // 4:height // 4 * 3, width // 4:width // 4 * 3, :]



        mask_whole[:,:,:]=mask_whole1[0:height1,0:width1,:]
        mask_whole_img = Image.fromarray(mask_whole)
        mask_whole_img.save(OutPath+listNameGroup[img_num][i][0:-8]+'.png', 'png')



if __name__ == '__main__':
    run()