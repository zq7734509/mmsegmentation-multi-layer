from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class POTSDAMDataset(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = (
        'build', 'car', 'grass', 'impervious', 'tree', )

    PALETTE = [[0,0,255],[255,255,0],[0,255,255],[255,255,255],[0,255,0]]

    def __init__(self, **kwargs):
        super(POTSDAMDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            # add input
            ir_img_suffix='.tif',
            dsm_img_suffix='.png',
            reduce_zero_label=True,
            **kwargs)
