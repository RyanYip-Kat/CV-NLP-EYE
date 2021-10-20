import os
import cv2
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm

def get_mask(img_file,res=5,outfile="test.png"):
    img = Image.open(img_file)
    img_array=np.asarray(img)
    img2gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
    ret, mask = cv2.threshold(img2gray, res, 255, cv2.THRESH_BINARY)
    img_mask = Image.fromarray(mask)

    img_mask.save(outfile)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",type=str)
    parser.add_argument("--res",type=int)
    parser.add_argument("--outdir",type=str)
    args = parser.parse_args()

    data_root_path = args.data
    outdir = args.outdir
    res = args.res

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    imgs = [x for x in os.listdir(data_root_path) if "jpg" in x]
    for img in tqdm(imgs,total=len(imgs)):
        filename = outdir +"/" + img.split(".")[0]+"_mask.png"
        img_file = data_root_path +"/" + img
        get_mask(img_file,res,filename)

    
