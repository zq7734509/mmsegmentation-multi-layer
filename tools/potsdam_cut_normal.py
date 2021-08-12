import os
from PIL import Image
from libtiff import TIFF
import numpy as np

# Potsdam标准数据集的制作, 各小块之间采用半重叠方式

def splitimage(src, rowheight, colwidth, dstpath):
    img = Image.open(src)
    w, h = img.size
    rowheight_half = rowheight // 2
    colwidth_half = colwidth // 2
    if rowheight <= h and colwidth <= w:
        print('Original image info: %sx%s, %s, %s' % (w, h, img.format, img.mode))
        print('waiting...')

        s = os.path.splitext(src)
        fn = s[0].split('/')
        basename = fn[-1]
        print('Original image name: %s' % basename)
        ext = 'png'

        num = 0
        rownum = h // rowheight_half - 1
        colnum = w // colwidth_half - 1
        for r in range(rownum):
            for c in range(colnum):
                box = (c * colwidth_half, r * rowheight_half, c * colwidth_half + colwidth, r * rowheight_half + rowheight)
                s = "%03d"%(num)
                cc = basename.split('_')
                basename = cc[0] + '_' + cc[1] + '_' + cc[2] + '_' + cc[3] + '_' + 'RGB'
                serial_name = basename + '_' + s
                if ext == 'tif':
                    tif_temp = TIFF.open(os.path.join(dstpath, serial_name + '.' + ext), mode='w')
                    tif_temp.write_image(np.array(img.crop(box)), compression=None)
                    tif_temp.close()
                else:
                    img.crop(box).save(os.path.join(dstpath, serial_name + '.' + ext), ext)
                # f.write(serial_name)
                # f.write('\n')
                num = num + 1

        print('total num: %s' % num)
    else:
        print('invalid')

def main_(file_dir):
    for root, dirs, files in os.walk(file_dir):
       for tempName in files:
             src = doc_path + '/' + tempName
             splitimage(src, 600, 600, dstpath)



doc_path ='/media/allen/orange/mm_dataset/potsdam_normal/ndsm_test'
dstpath = '/media/allen/orange/mm_dataset/mydataset_Potsdam_600_normal/ndsm_dir/val'
# f = open('/media/allen/orange/mm_dataset/vaihingen_600_new/img_dir/val.txt', 'a')
# f.truncate()

main_(doc_path)

# f.close()