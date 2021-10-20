import os
import random
import cv2
import numpy as np
import argparse
import json
from PIL import Image
from tqdm import tqdm
import torch.utils.data as data
import matplotlib.pyplot as plt
def get_lesion_labels(data_path):
    labels = [x for x in os.listdir(data_path) if "json" in x]
    images= [x.split(".")[0]+".jpg" for x in labels]
    print("#nImages : {}".format(len(images)))
    
    lesion_label_list=[]
    lesion_label_dict={}
    for i in range(len(labels)):
        lab_path = data_path+"/"+ labels[i]
        label = json.load(open(lab_path, "r"))
        lesion_label_dict[labels[i]]=[]
        for idx, lesion_label in enumerate(label["Lesion"].keys()):
            lesion_label_dict[labels[i]].append(lesion_label)
            #if lesion_label not in lesion_label_list:
            lesion_label_list.append(lesion_label)
    
    lesion_label2idx={x:i for i,x in enumerate(np.unique(lesion_label_list))}
    return lesion_label2idx,lesion_label_dict,lesion_label_list

class YanQDLesionDisease(data.Dataset):
    def __init__(self,args,split="train"):
        super(YanQDLesionDisease,self).__init__
        self.args = args
        self.image_path = self.args.root_path
        self.mode = split
        self.disease =  args.disease 
        self.image_size = self.args.img_size
        self.shuffle = True if split == "train" else False
        
        self.lable2idx,self.label_info,self.disease_label = get_lesion_labels(self.image_path+"/" + self.mode)
        self.labels = list(self.label_info.keys())
        self.images = [k.split(".")[0]+".jpg" for k in self.labels]
        self.color_lists=  {lab:(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for lab in self.lable2idx.keys()}
        self.num_classes = len(self.lable2idx)
        self.update()
        print("#nImage : {}".format(self.__len__()))
    
    def update(self):
        if self.disease is not None:
            labels = []
            for labj,labd in self.label_info.items():
                if self.disease in labd:
                    labels.append(labj)
            images = [k.split(".")[0]+".jpg" for k in labels]
            self.images = images
            self.labels = labels
            labs=[]
            for lab in self.labels:
                labs+=self.label_info[lab]
            self.num_classes= len(np.unique(labs))
            self.lable2idx = {lab:i for i,lab in enumerate(np.unique(labs))}
            self.label_info = [self.label_info[lab] for lab in self.labels]
                
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = self.image_path+"/"+ self.mode + "/" + self.images[index]
        lab_path = self.image_path+"/"+ self.mode + "/" + self.labels[index]
        image = cv2.imread(img_path)
        label = json.load(open(lab_path, "r"))

        mask = np.zeros((image.shape[0],image.shape[1],self.num_classes))
        mask2 = np.zeros((image.shape[0],image.shape[1]))
        for idx, lesion_label in enumerate(label["Lesion"].keys()):
                msk=np.zeros((image.shape[0],image.shape[1]))
                pts = [np.asarray(item) for item in label["Lesion"][lesion_label]]
                cv2.fillPoly(msk, pts,color = self.color_lists[lesion_label])
                cv2.fillPoly(mask2, pts,color = self.color_lists[lesion_label])
                msk[msk>0] = self.lable2idx[lesion_label]+1
                #msk[msk>0] = idx+1
                mask2 = np.maximum(mask2,msk)
                mask[:,:,self.lable2idx[lesion_label]] = msk
        
        img =  image.copy()
        if self.image_size is not None:
            img =  cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size))
            mask2 = cv2.resize(mask2, (self.image_size, self.image_size))
        mask_all = mask.copy()
        #img =  np.transpose(img, (2,0,1)).astype(np.float32)
        if self.disease is not None:
            mask = mask[:,:,self.lable2idx[self.disease]]
        else:
            mask = np.argmax(mask,axis=-1)

        case_name = self.images[index].split(".")[0] 
        return img,mask,mask_all,mask2,case_name


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path",type=str)
    parser.add_argument("--img_size",type=int,default=None)
    parser.add_argument("--outdir",type=str)
    parser.add_argument("--split",type=str,default="train",choices=["train","val","test"])
    parser.add_argument("--disease",type=str,default="Cataract",help="which disease")
    args = parser.parse_args()

    outdir = args.outdir

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    dataset = YanQDLesionDisease(args,split=args.split)
    for i in tqdm(range(len(dataset)),total=len(dataset)):
        image,mask,mask_all,mask2,case_name=dataset[i]
        msk = Image.fromarray(mask.astype(np.uint8))
        filename = args.outdir+"/"+ case_name +"_anno.png"
        #msk.save(filename)
        plt.imsave(filename,mask,cmap="gray")

    
