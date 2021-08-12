#coding:utf-8
import os
from PIL import Image
import numpy as np
#源目录
MyPath = '/media/allen/orange/00第二篇论文实验结果/upernet_swin_base_potsdam_normal_240k/small_img/'
#输出目录
OutPath = '/media/allen/orange/00第二篇论文实验结果/upernet_swin_base_potsdam_normal_240k/rgb_img/'
width=600
height=600
width1=6000
height1=6000
width_num=width1 // width
height_num=height1 // height

def splitList(list_all):
    list_unique = []
    list_name_group = []
    for i in range(len(list_all)):
        ss = list_all[i].split('_')
        if ss[2] not in list_unique:
            list_unique.append(ss[2])
            list_name_group.append([])
            list_name_group[-1].append(list_all[i])
        else:
            list_name_group[list_unique.index(ss[2])].append(list_all[i])

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
    for img_num in range(len(listNameGroup)):
        for i in range(width_num):
            for j in range(height_num):

                print('name: %s' % listNameGroup[img_num][i*width_num+j])
                img = Image.open(MyPath + listNameGroup[img_num][i*width_num+j])
                img_array = np.asarray(img)
                mask_whole1[i*width:i*width+width,j*width:j*width+width,:] = img_array[:,:,:]

        mask_whole[:,:,:]=mask_whole1[0:height1,0:width1,:]
        mask_whole_img = Image.fromarray(mask_whole)
        mask_whole_img.save(OutPath+listNameGroup[img_num][i*width_num+j][0:-8], 'png')



if __name__ == '__main__':
    run()