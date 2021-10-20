import matplotlib.pyplot as plt
import numpy as np

def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()

def concat_img_and_mask(img, mask,gt=None,outfile="test.png"):
    N = 2
    if gt is not None:
        N = 3
    fig, ax = plt.subplots(1, N,figsize=(16,16))
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title(f'Output mask')
    ax[1].imshow(mask)
    if gt is not None:
        ax[2].set_title(f'GrundTruth')
        ax[2].imshow(gt)
    #plt.xticks([]), plt.yticks([])
    #plt.show()
    plt.savefig(outfile)

def add_mask_to_image(img,mask):
    msk = np.asarray(mask).astype(np.int32)
    image = np.asarray(img).astype(np.int32)
    if len(msk.shape)==2:
        mask1d=msk[:,:,np.newaxis]
        mask3d=np.concatenate((mask1d,mask1d,mask1d),axis=-1).astype(np.int32)
    elif len(msk.shape)==3 and msk.shape[2] ==3:
        mask3d =msk.copy()
    else:
        raise ValueError("Invalid mask")

    return image+mask3d


def concat_result(ori_img,pred_res,gt):
    ori_img = data = np.transpose(ori_img,(1,2,0))
    pred_res = data = np.transpose(pred_res,(1,2,0))
    gt = data = np.transpose(gt,(1,2,0))

    binary = deepcopy(pred_res)
    binary[binary>=0.5]=1
    binary[binary<0.5]=0

    if ori_img.shape[2]==3:
        pred_res = np.repeat((pred_res*255).astype(np.uint8),repeats=3,axis=2)
        binary = np.repeat((binary*255).astype(np.uint8),repeats=3,axis=2)
        gt = np.repeat((gt*255).astype(np.uint8),repeats=3,axis=2)
    total_img = np.concatenate((ori_img,pred_res,binary,gt),axis=1)
    return total_img

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

import colorsys
import random
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
