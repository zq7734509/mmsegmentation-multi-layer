from PIL import Image
import numpy as np
from libtiff import TIFF
img = Image.open('/media/allen/orange/mm_dataset/mydataset_Potsdam_600_normal/ndsm_dir/train/top_potsdam_02_10_RGB_000.png')
print(img.mode)
k = np.array(img)
h,w = k.shape
pad = k[:,-1]
pad = pad[:,np.newaxis];
c = np.concatenate((k,pad),axis=1)
tif = TIFF.open('/media/allen/orange/mm_dataset/potsdam_normal/dsm_test/top_potsdam_03_13_dsm_new.tif', mode='w')
tif.write_image(c, compression=None)
tif.close()
a = 1
# tif = TIFF.open('/home/allen/img1.tif', mode='w')
# tif.write_image(k, compression=None)
# tif.close()
