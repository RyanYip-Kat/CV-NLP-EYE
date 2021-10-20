import os.path as osp
from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()  # 注册   不要忘记在__init__.py作显示导入
class YanQDStructure(CustomDataset):
    CLASSES = ('background', 'Eyelid', 'Cornea', 'Pupil')   # 类别名称设置
    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156]]  # 调色板设置

    def __init__(self,**kwargs):
        super(YanQDStructure, self).__init__(
            img_suffix='.jpg',  # img文件‘后缀’
            seg_map_suffix='_mask.png',  # gt文件‘后缀’
            #reduce_zero_label=True,  # num_classes ==2,False, else True 
            **kwargs)
        assert osp.exists(self.img_dir)
