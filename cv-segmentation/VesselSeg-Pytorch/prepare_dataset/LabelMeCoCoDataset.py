import json
import os
import numpy as np

import pycocotools
import torch
import skimage.io
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from torch.utils.data import Dataset

class LabelMeCOCO(Dataset):
    def __init__(self,image_path,json,catId=None,transform=None):
        super(LabelMeCOCO,self).__init__
        self.image_path = image_path
        self.anntation_file = json
        self.transform =transform
        self.coco = COCO(self.anntation_file)
        self.catId = catId

        self.image_ids = self.coco.getImgIds()
        self.catIds = self.coco.getCatIds()
        self.load_classes()

    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        # coco ids is not from 1, and not continue
        # make a new index from 0 to 79, continuely

        # classes:             {names:      new_index}
        # coco_labels:         {new_index:  coco_index}
        # coco_labels_inverse: {coco_index: new_index}
        self.classes, self.coco_labels, self.coco_labels_inverse = {}, {}, {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']]   = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # labels:              {new_index:  names}
        self.labels = {}
        for k, v in self.classes.items():
            self.labels[v] = k

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img = self.load_image(index)
        ann = self.load_anns(index)
        sample = {'img':img, 'ann': ann}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, index):
        image_info = self.coco.loadImgs(self.image_ids[index])[0]
        imgpath =  os.path.join(self.image_path,image_info['file_name'])

        img = skimage.io.imread(imgpath)
        return img.astype(np.float32) / 255.0

    def load_anns(self, index):
        annotation_ids = self.coco.getAnnIds(self.image_ids[index], catIds=self.catId,iscrowd=False)
        anns = self.coco.loadAnns(annotation_ids)
        mask = self.coco.annToMask(anns[0])
        for i in range(len(anns)):
            mask += self.coco.annToMask(anns[i])
        return mask.astype(np.float32)

    def image_aspect_ratio(self, index):
        image = self.coco.loadImgs(self.image_ids[index])[0]
        return float(image['width']) / float(image['height'])


if __name__=="__main__":
    import argparse
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str,default=None,help="image_path")
    parser.add_argument('--jsonfile', type=str,default=None,help="coco json file from labelme2coco.py")
    parser.add_argument('--catid',type=int,default=None,nargs="+",help="catid use to return specify annotation mask file")
    parser.add_argument('--outdir',type=str,default="annotations",help="path to save annotations")
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    dataset = LabelMeCOCO(args.image_path,args.jsonfile,args.catid)
    for i in tqdm(range(len(dataset)),total=len(dataset)):
        sample=dataset[i]
        img,mask=sample["img"],sample["ann"]
        msk = Image.fromarray(mask.astype(np.uint8))
        image_info = dataset.coco.loadImgs(dataset.image_ids[i])[0] 
        img_file = os.path.basename(image_info["file_name"])
        filename = args.outdir +"/" + img_file.split(".")[0]+"_ann.png"
        plt.imsave(filename,msk,cmap="gray")


