import argparse
import logging
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask,concat_img_and_mask,add_mask_to_image
from utils.utils import apply_mask,random_colors
def predict_img(net,
                full_img,
                img_size,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    if type(full_img)==np.ndarray:
        img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False,img_size=img_size))
    elif type(full_img)==torch.Tensor:
        img = full_img
    else:
        raise ValueError("Invalid image")

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()

def find_high_ligth(image,ligth_res=254):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray2 = img_gray > ligth_res
    return img_gray2.astype(np.int32)

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    #parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    #parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of output images')
    #parser.add_argument("--test_data",type=str,default=None)
    parser.add_argument("--outdir",type=str,default="predictions")
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument("--img_w",type=int,default=512)
    parser.add_argument("--img_h",type=int,default=512)

    parser.add_argument("--test_image_dir",type=str,default=None,help="image dir ")
    parser.add_argument("--test_mask_dir",type=str,default=None,help="mask dir")
    parser.add_argument("--mask_suffix",type=str,default=None)
    parser.add_argument('--rm_hl',action='store_true',help="wether mask hight light area on image")
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    args = get_args()
    args.img_size = [args.img_w,args.img_h]
    #in_files = args.input
    #out_files = get_output_filenames(args)
    outdir = args.outdir
    #test_data = args.test_data
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    #in_files = [x  for x in os.listdir(test_data) if "jpg" in x or "jpeg" in x]
    test_dir_img = Path(args.test_image_dir)
    test_dir_mask = Path(args.test_mask_dir)

    dataset = BasicDataset(test_dir_img, test_dir_mask, args.scale,args.mask_suffix,args.img_size)

    net = UNet(n_channels=3, n_classes=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')
    #colors = random_colors(10)
    colors=[(0.0, 1.0, 0.40000000000000036),(1.0, 0.0, 0.0),(0.8000000000000007, 0.0, 1.0),(0.0, 0.40000000000000036, 1.0),(0.7999999999999998, 1.0, 0.0)]
    #for i, filename in enumerate(in_files):
    #    logging.info(f'\nPredicting image {filename} ...')
    #    img = Image.open(test_data +"/"+ filename)
    #    image = img.resize((int(args.img_w * args.scale),int(args.img_h * args.scale)))
    for i in tqdm(range(len(dataset)),total=len(dataset)):
        sample = dataset[i]
        img,gt = sample["image"],sample["mask"]

        try:
            image = Image.open(os.path.join(test_dir_img,dataset.names[i]+".jpeg"))
        except:
            image = Image.open(os.path.join(test_dir_img,dataset.names[i]+".jpg"))
        image = image.resize((int(args.img_w * args.scale),int(args.img_h * args.scale)))
        gt = gt.detach().numpy().astype(np.int32)
        mask = predict_img(net=net,
                           full_img=img,
                           img_size = args.img_size,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        filename = dataset.names[i]
        if args.rm_hl:
            hl_image_seg = find_high_ligth(np.asarray(image).astype(np.uint8),254)
            hl_image_seg[hl_image_seg>1] = 1
        if not args.no_save:
            #out_filename = outdir +"/" + filename.split(".")[0]+"_mask.png"
            result = mask_to_image(mask)
            res = np.array(result).astype(np.int32)
            res[res>1] =1

            gt[gt>1] =1 
            #result.save(out_filename)
            #mutiplt_img = add_mask_to_image(image,result)
            #mutiplt_img2 = add_mask_to_image(image,gt)

            #out_filename1 = outdir +"/" + filename.split(".")[0]+"_pred1.png"
            #concat_img_and_mask(image,result,out_filename1) 
            imm1 = apply_mask(np.asarray(image).astype(np.uint8),res,colors[1])
            imm2 = apply_mask(np.asarray(image).astype(np.uint8),gt,colors[0])
            if args.rm_hl:
                imm1 = apply_mask(imm1.astype(np.uint8),hl_image_seg,colors[2])
                imm2 = apply_mask(imm2.astype(np.uint8),hl_image_seg,colors[2])

            out_filename = outdir +"/" + filename.split(".")[0]+"_pred.png"
            concat_img_and_mask(image,imm1,imm2,out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(image, mask)
