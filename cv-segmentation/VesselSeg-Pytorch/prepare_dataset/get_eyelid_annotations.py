import os
import random
import numpy as np
import torch
import cv2
import json
import torch.utils.data as data

from PIL import Image

class YanQDEyelid(data.Dataset):
    def __init__(self,args,split="train",structure="Eyelid"):
        super(YanQDEyelid,self).__init__
        self.args = args
        self.image_path = self.args.root_path
        self.mode = split
        self.structure = structure
        self.image_size = self.args.img_size
        self.shuffle = True if split == "train" else False

        self.labels = [x for x in os.listdir(self.image_path+"/"+self.mode) if "json" in x]
        self.images = [k.split(".")[0]+".jpg" for k in self.labels]
        self.color_lists=  {lab:(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for lab in ["Eyelid", "Cornea", "Pupil"]}
        self.lable2idx = {lab:i for i,lab in enumerate(["Eyelid", "Cornea", "Pupil"])}

        self.num_classes = len(self.lable2idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.image_path+"/"+ self.mode + "/" + self.images[index]
        lab_path = self.image_path+"/"+ self.mode + "/" + self.labels[index]
        image = cv2.imread(img_path)
        label = json.load(open(lab_path, "r"))

        mask = np.zeros((image.shape[0],image.shape[1],len(self.lable2idx)))
        mask2 = np.zeros((image.shape[0],image.shape[1]))
        for idx, structure_label in enumerate(["Eyelid", "Cornea", "Pupil"]):
                #msk=np.zeros((image.shape[0],image.shape[1]))
                if structure_label in label["Structure"]:
                    msk=np.zeros((image.shape[0],image.shape[1]))
                    str_array=np.asarray(label["Structure"][structure_label])
                    str_color = self.color_lists[structure_label]
                    cv2.fillPoly(msk,[str_array],color=str_color)
                    msk[msk>0] = idx+1
                    #mask2 = mask2+msk
                    mask2 = np.maximum(mask2,msk)
                    mask[:,:,self.lable2idx[structure_label]] = msk
                #cv2.fillPoly(msk, [np.asarray(label["Structure"][structure_label])], color = self.color_lists[structure_label])
                #msk[msk>0] = idx+1
                #mask[:,:,self.lable2idx[structure_label]] = msk

        if self.structure is not None:
            assert self.structure in self.lable2idx
            label = mask[:,:,self.lable2idx[self.structure]]
        else:
            label = np.argmax(mask,axis=-1).astype(np.float32)

        if self.image_size is not None:
            image =  cv2.resize(image, (self.image_size, self.image_size))
            #img =  np.transpose(img, (2,0,1)).astype(np.float32)
            mask = cv2.resize(mask, (self.image_size, self.image_size))
            label = cv2.resize(label,(self.image_size, self.image_size))
            mask2 = cv2.resize(mask2, (self.image_size, self.image_size))

        #sample = {'image': img, 'label': msk_max,"mask" : msk}
        case_name = self.labels[index].split(".")[0]
        return image,label,mask,mask2,case_name

if __name__=="__main__":
    import argparse
    import os
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path",type=str)
    parser.add_argument("--img_size",type=int,default=None)
    parser.add_argument("--outdir",type=str)
    parser.add_argument("--split",type=str,default="train",choices=["train","val","test"])
    parser.add_argument("--structure",type=str,default="Eyelid",choices=["Eyelid", "Cornea", "Pupil"],help="which disease")
    args = parser.parse_args()

    outdir = args.outdir

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    dataset = YanQDEyelid(args,split=args.split,structure=args.structure)
    for i in tqdm(range(len(dataset)),total=len(dataset)):
        try:
            img,str_label,str_mask,str_mask2,case_name = dataset[i]
            #msk = Image.fromarray(mask.astype(np.uint8))
            filename = outdir+"/"+ case_name +"_anno.png"
            #msk.save(filename)
            #plt.imsave(filename,mask,cmap="gray")
            cv2.imwrite(filename,str_label)
            filename = outdir+"/"+ case_name +"_mask.png"
            #str_mask_gray = cv2.cvtColor(str_mask.astype(np.uint8),cv2.COLOR_BGR2GRAY)
            cv2.imwrite(filename,str_mask2)
        except:
            print("Invalid annotation {}".format(dataset.labels[i]))
            continue
