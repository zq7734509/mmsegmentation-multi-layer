import os
from PIL import Image
import pdb

def splitimage(src, rowheight, colwidth, dstpath):
    img = Image.open(src)
    w, h = img.size
    if rowheight <= h and colwidth <= w:
        print('Original image info: %sx%s, %s, %s' % (w, h, img.format, img.mode))
        print('waiting...')

        s = os.path.splitext(src)
        fn = s[0].split('/')
        basename = fn[-1]
        #ext = fn[-1]
        print('Original image name: %s' % basename)
        ext = 'png'

        num = 0
        rownum = h // rowheight
        colnum = w // colwidth
        for r in range(rownum):
            for c in range(colnum):
                box = (c * colwidth, r * rowheight, (c + 1) * colwidth, (r + 1) * rowheight)
                s = "%03d"%(num)
                cc = basename.split('_')
                basename = cc[0] + '_' + cc[1] + '_' + cc[2] + '_' + cc[3] + '_' + 'RGB'
                serial_name = basename + '_' + s
                img.crop(box).save(os.path.join(dstpath, serial_name + '.' + ext), ext)
                f.write(serial_name)
                f.write('\n')
                num = num + 1
            # if (w % colwidth) > 0.5 * colwidth:
            if (w % colwidth) > 0:
                box = (w - colwidth, r * rowheight, w, (r + 1) * rowheight)
                s = "%03d" % (num)
                cc = basename.split('_')
                basename = cc[0] + '_' + cc[1] + '_' + cc[2] + '_' + cc[3] + '_' + 'RGB'
                serial_name = basename + '_' + s
                img.crop(box).save(os.path.join(dstpath, serial_name + '.' + ext), ext)
                f.write(serial_name)
                f.write('\n')
                num = num + 1
        # if (h % rowheight) > 0.5 * rowheight:
        if (h % rowheight) > 0:
            for c in range(colnum):
                box = (c * colwidth, h - rowheight, (c + 1) * colwidth, h)
                s = "%03d" % (num)
                cc = basename.split('_')
                basename = cc[0] + '_' + cc[1] + '_' + cc[2] + '_' + cc[3] + '_' + 'RGB'
                serial_name = basename + '_' + s
                img.crop(box).save(os.path.join(dstpath, serial_name + '.' + ext), ext)
                f.write(serial_name)
                f.write('\n')
                num = num + 1
            # if (w % colwidth) > 0.5 * colwidth:
            if (w % colwidth) > 0:
                box = (w - colwidth, h - rowheight, w, h)
                s = "%03d" % (num)
                cc = basename.split('_')
                basename = cc[0] + '_' + cc[1] + '_' + cc[2] + '_' + cc[3] + '_' + 'RGB'
                serial_name = basename + '_' + s
                img.crop(box).save(os.path.join(dstpath, serial_name + '.' + ext), ext)
                f.write(serial_name)
                f.write('\n')
                num = num + 1
        print('total num: %s' % num)
    else:
        print('invalid')

def main_(file_dir):
    for root, dirs, files in os.walk(file_dir):
       for tempName in files:
             src = doc_path + '/' + tempName
             splitimage(src, 600, 600, dstpath)

doc_path ='/media/allen/orange/ISPRS dataset/Vaihingen/allen/label/val'
dstpath = '/media/allen/orange/mm_dataset/vaihingen_600_new/ann_dir/val'
f = open('/media/allen/orange/mm_dataset/vaihingen_600_new/ann_dir/val.txt', 'a')
f.truncate()

main_(doc_path)

f.close()