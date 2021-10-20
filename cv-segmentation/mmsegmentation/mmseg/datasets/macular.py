import os.path as osp
from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()  # 注册   不要忘记在__init__.py作显示导入
class MacularDataset(CustomDataset):
    CLASSES = ('background', 'macular')   # 类别名称设置
    PALETTE = [[120, 120, 120], [6, 230, 230]]  # 调色板设置

    def __init__(self,**kwargs):
        super(MacularDataset, self).__init__(
            img_suffix='.jpeg',  # img文件‘后缀’
            seg_map_suffix='_ann.png',  # gt文件‘后缀’
            reduce_zero_label=False,  # num_classes ==2,False, else True 
            **kwargs)
        assert osp.exists(self.img_dir)
