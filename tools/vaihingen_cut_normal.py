import os
from PIL import Image
from libtiff import TIFF
import numpy as np

# Vaihingen标准数据集的制作, 各小块之间采用半重叠方式

srcPath ='/media/allen/orange/mm_dataset/vaihingen_normal/ndsm_test'
dstPath = '/media/allen/orange/mm_dataset/vaihingen_600_normal/ndsm_dir/val'


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
        row_need_padding = False
        col_need_padding = False
        rownum = h // rowheight_half
        colnum = w // colwidth_half
        if h > rownum * rowheight_half:
            row_need_padding = True
        if w > colnum * colwidth_half:
            col_need_padding =True

        # 先行后列
        for r in range(rownum-1):
            for c in range(colnum-1):
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
                num = num + 1
            if col_need_padding:
                box = (w-colwidth, r * rowheight_half, w, r * rowheight_half + rowheight)
                s = "%03d" % (num)
                cc = basename.split('_')
                basename = cc[0] + '_' + cc[1] + '_' + cc[2] + '_' + cc[3] + '_' + 'RGB'
                serial_name = basename + '_' + s
                if ext == 'tif':
                    tif_temp = TIFF.open(os.path.join(dstpath, serial_name + '.' + ext), mode='w')
                    tif_temp.write_image(np.array(img.crop(box)), compression=None)
                    tif_temp.close()
                else:
                    img.crop(box).save(os.path.join(dstpath, serial_name + '.' + ext), ext)
                num = num + 1
        if row_need_padding:
            for c in range(colnum-1):
                box = (c * colwidth_half, h-rowheight, c * colwidth_half + colwidth, h)
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
                num = num + 1
            if col_need_padding:
                box = (w-colwidth, h-rowheight, w, h)
                s = "%03d" % (num)
                cc = basename.split('_')
                basename = cc[0] + '_' + cc[1] + '_' + cc[2] + '_' + cc[3] + '_' + 'RGB'
                serial_name = basename + '_' + s
                if ext == 'tif':
                    tif_temp = TIFF.open(os.path.join(dstpath, serial_name + '.' + ext), mode='w')
                    tif_temp.write_image(np.array(img.crop(box)), compression=None)
                    tif_temp.close()
                else:
                    img.crop(box).save(os.path.join(dstpath, serial_name + '.' + ext), ext)
                num = num + 1
        print('total num: %s' % num)
    else:
        print('invalid')

if __name__ == '__main__':
    for root, dirs, files in os.walk(srcPath):
       for tempName in files:
             src = srcPath + '/' + tempName
             splitimage(src, 600, 600, dstPath)


