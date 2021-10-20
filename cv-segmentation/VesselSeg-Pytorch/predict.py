import PIL
import joblib,copy
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch,sys
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict
from lib.visualize import save_img,group_images,concat_result
import os
import argparse
from lib.logger import Logger, Print_Logger
from lib.extract_patches import *
from os.path import join
from lib.dataset import TestDataset
from lib.metrics import Evaluate
import models
from lib.common import setpu_seed,dict_round
from config import parse_args
from lib.pre_processing import my_PreProc

setpu_seed(2021)

import matplotlib.pyplot as plt
def save_pred_single(pred_images,image_names,save_path):
    for i,img in enumerate(pred_images):
        img=np.squeeze(np.transpose(img,(1,2,0)),axis=2)
        plt.imsave(save_path+"/"+image_names[i]+"_pred.png",img,cmap="gray")
        #img = Image.fromarray(img.astype(np.uint8))
        #img.save(save_path+"/"+image_names[i]+"_pred.png")


def concat_result2(ori_img,pred_res):
    from copy import deepcopy
    ori_img = data = np.transpose(ori_img,(1,2,0))
    pred_res = data = np.transpose(pred_res,(1,2,0))

    binary = deepcopy(pred_res)
    binary[binary>=0.5]=1
    binary[binary<0.5]=0

    if ori_img.shape[2]==3:
        pred_res = np.repeat((pred_res*255).astype(np.uint8),repeats=3,axis=2)
        binary = np.repeat((binary*255).astype(np.uint8),repeats=3,axis=2)
    total_img = np.concatenate((ori_img,pred_res,binary),axis=1)
    return total_img

def readImg(im_fn,img_size=None):
    """
    When reading local image data, because the format of the data set is not uniform,
    the reading method needs to be considered. 
    Default using pillow to read the desired RGB format img
    """
    img = PIL.Image.open(im_fn)
    if img_size is not None:
        img = img.resize((img_size,img_size))
    return img


def load_images(img_list,img_size,patch_height, patch_width, stride_height, stride_width):
    imgs = None
    for i in range(len(img_list)):
        img = np.asarray(readImg(img_list[i],img_size))
        if len(img.shape)==2:
            img = np.expand_dims(img,axis=2)
            img = np.repeat(img,repeats=3,axis=2)
        imgs = np.expand_dims(img,0) if imgs is None else np.concatenate((imgs,np.expand_dims(img,0)))
    
    #Convert the dimension of imgs to [N,C,H,W]
    imgs = np.transpose(imgs,(0,3,1,2))
    print('ori data shape < ori_imgs:{}'.format(imgs.shape))
    print("imgs pixel range %s-%s: " %(str(np.min(imgs)),str(np.max(imgs))))
    print("==================data have loaded======================")
    
    test_imgs = my_PreProc(imgs)
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)
    print("\nTest images shape: {}, vaule range ({} - {}):"\
        .format(test_imgs.shape, str(np.min(test_imgs)), str(np.max(test_imgs))))
    
    patches_imgs_test = extract_ordered_overlap(test_imgs,patch_height,patch_width,stride_height,stride_width)
    print("test patches shape: {}, value range ({} - {})"\
        .format(patches_imgs_test.shape, str(np.min(patches_imgs_test)), str(np.max(patches_imgs_test))))
    return patches_imgs_test,imgs,test_imgs.shape[2], test_imgs.shape[3]

class Predictor():
    def __init__(self, args,img_list,img_size=None):
        self.args = args
        self.img_list = img_list
        self.img_size = img_size
        self.img_name_list = [os.path.basename(x).split(".")[0] for x in img_list]
        assert (args.stride_height <= args.test_patch_height and args.stride_width <= args.test_patch_width)
        # save path
        self.path_experiment = join(args.outf, args.save)

        self.patches_imgs_test, self.test_imgs, self.new_height, self.new_width = load_images(
            img_list= self.img_list,
            img_size= self.img_size,
            patch_height=args.test_patch_height,
            patch_width=args.test_patch_width,
            stride_height=args.stride_height,
            stride_width=args.stride_width
        )
        self.img_height = self.test_imgs.shape[2]
        self.img_width = self.test_imgs.shape[3]

        test_set = TestDataset(self.patches_imgs_test)
        self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=3)

    # Inference prediction process
    def inference(self, net):
        net.eval()
        preds = []
        with torch.no_grad():
            for batch_idx, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                inputs = inputs.cuda()
                outputs = net(inputs)
                outputs = outputs[:,1].data.cpu().numpy()
                preds.append(outputs)
        predictions = np.concatenate(preds, axis=0)
        self.pred_patches = np.expand_dims(predictions,axis=1)
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
        ## restore to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]
        return self.pred_imgs

    def save_segmentation_result(self,save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # self.test_imgs = my_PreProc(self.test_imgs) # Uncomment to save the pre processed image
        save_pred_single(self.pred_imgs,self.img_name_list,save_path)
        for i in range(self.test_imgs.shape[0]):
            total_img = concat_result2(self.test_imgs[i],self.pred_imgs[i])
            save_img(total_img,join(save_path, "Result_"+self.img_name_list[i]+'.png'))
    def save_pred_npy(self,save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save('{}/result.npy'.format(save_path),self.pred_imgs)


def get_config():
    import argparse
    parser = argparse.ArgumentParser()

    # in/out
    parser.add_argument('--outf', default='./experiments',
                        help='trained model will be saved at here')
    parser.add_argument('--save', default='UNet_vessel_seg',
                        help='save name of experiment in args.outf directory')
    
    parser.add_argument('--batch_size', default=64,
                        type=int, help='batch size')
    # testing
    parser.add_argument("--test_data",type=str,default=None)
    parser.add_argument('--test_patch_height', default=96,type=int)
    parser.add_argument('--test_patch_width', default=96,type=int)
    parser.add_argument('--stride_height', default=16,type=int)
    parser.add_argument('--stride_width', default=16,type=int)
    parser.add_argument('--img_size',default=None,type=int)

    # hardware setting
    parser.add_argument('--device', default=3, type=int,
                        help='Use GPU calculating')

    parser.add_argument('--outdir',default="result",type=str)
    args = parser.parse_args()
    return args

        
if __name__ == '__main__':
    args = get_config()
    save_path = join(args.outf, args.save)
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    # net = models.UNetFamily.Dense_Unet(1,2).to(device)
    net = models.LadderNet(inplanes=1, num_classes=2, layers=3, filters=16).to(device)
    cudnn.benchmark = True

    # Load checkpoint
    print('==> Loading checkpoint...')
    checkpoint = torch.load(join(save_path, 'best_model.pth'))
    net.load_state_dict(checkpoint['net'])
    
    img_list = [ args.test_data +"/"+f for f in os.listdir(args.test_data) if "jpg" in f or "jpeg" in f ] 
    img_size = args.img_size
    save_path = args.outdir

    predictor = Predictor(args,img_list,img_size)
    pred_imgs=predictor.inference(net)
    
    if not os.path.exists(save_path):
            os.makedirs(save_path)

    np.save('{}/result.npy'.format(save_path),pred_imgs)
    predictor.save_segmentation_result(save_path)
