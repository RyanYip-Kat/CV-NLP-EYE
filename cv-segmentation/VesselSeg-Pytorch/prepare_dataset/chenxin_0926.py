import os
import numpy as np
import matplotlib.pyplot as plot
from PIL import Image
from os.path import join
from tqdm import tqdm

def get_data(data_path,outdir="data_path_list/YanQD"):
    anno_path = join(data_path,"annotations")
    img_path = join(data_path,"images")
    mask_path = join(data_path,"masks")
    
    anno_files = os.listdir(anno_path)
    anno_names = [ x.split("_HE")[0]  for x in anno_files]
    mask_files = [x+"_mask.png" for x in anno_names]
    img_files =[x+".jpg" for x in anno_names]
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    outfile = outdir +"/"+"train.txt"
    with open(outfile,"w") as fout:
        for i in tqdm(range(len(anno_files)),total=len(anno_files)):
            img = img_path +"/"+img_files[i]
            msk = mask_path +"/"+mask_files[i]
            anno = anno_path+"/"+anno_files[i]
            line = " ".join([img,anno,msk])+"\n"
            fout.write(line)
    fout.close()

    
def preprocess_yanqd_annotations(data_path,split="train",disease="EX"):
    disease_dict={"HE":"Haemorrhages","EX":"HardExudates","MA":"Microaneurysms","OD":"OpticDisc","SE":"SoftExudates"}
    outdir  = "./annotation_gray"+"/"+split+"/"+disease_dict[disease]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    anno_path = anno_path = join(data_path,split,disease_dict[disease])
    anno_files = os.listdir(anno_path)
    for anno in tqdm(anno_files,total=len(anno_files)):
        out_anno = anno.split(".")[0]+".png"
        anno_img=Image.open(join(anno_path,anno))
        anno_array=np.asarray(anno_img)
        plt.imsave(join(outdir,out_anno),anno_array,cmap="gray")
        

if __name__=="__main__":
    data_path = "/path/VesselSeg-datasets/CHENXIN_HE_annotation/"
    get_data(data_path,"data_path_list/HE/")
