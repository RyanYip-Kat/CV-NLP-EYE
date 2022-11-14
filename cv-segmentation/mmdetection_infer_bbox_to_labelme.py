import mmcv
import os
import argparse
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector


import numpy as np
import base64
import PIL
import io
import json
import codecs

def encodeImageForJson(image):
    img_pil = PIL.Image.open(image)
    f = io.BytesIO()
    img_pil.save(f, format='PNG')
    data = f.getvalue()
    encData = codecs.encode(data, 'base64').decode()
    encData = encData.replace('\n', '')
    return encData

def write4json(session_dict,session_file):
    with open(session_file, 'w',encoding='utf-8') as f:
       json.dump(session_dict, f,ensure_ascii=False,indent=4)
    f.close()

def inferTolabelme(result,img_path,score_thr=0.3):

    bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
            ]
    labels = np.concatenate(labels)
    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
    ##############   format labelme
    shapes=[]
    image_labelme={}
    image_labelme["version"]="4.5.9"
    image_labelme["flags"]={}

    for i, bbox in enumerate(bboxes):
        label = str(labels[i])

        bbox_int = bbox.astype(np.int32)
    #poly=[[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
    #            [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        points=[[float(bbox_int[0]), float(bbox_int[1])],[float(bbox_int[2]), float(bbox_int[3])]]
        group_id=None
        fill_color=None
        shape_type="rectangle"
        flags={}
        shapes.append({"fill_color":fill_color,"label":label,"points":points,"group_id":group_id,"shape_type":shape_type,"flags":flags})
    image_labelme['shapes']=shapes
    image_labelme["imagePath"]=os.path.basename(img_path)
    image_labelme["imageHeight"]=860
    image_labelme["imageWidth"]=2000
    image_labelme["imgAttrLabel"]=None
    image_labelme["imageData"] = encodeImageForJson(img_path)
    return image_labelme

  if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_dir",type=str,default=None,help="test data image path")
    parser.add_argument("--score_thr",type=float,default=0.3,help="score thresold")
    parser.add_argument("--outdir",type=str,default="result")

    parser.add_argument("config",type=str,default=None,help="config file:configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py")
    parser.add_argument("checkpoint",type=str,default=None,help="checkpoint file:latest.pth")
    args = parser.parse_args()

    data_path = args.test_data_dir
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Specify the path to model config and checkpoint file
    config_file = args.config #'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = args.checkpoint #'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    score_thr=args.score_thr

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:7')

    img_files = [f for f in os.listdir(data_path)]
    for img in tqdm(img_files,total=len(img_files)):
        #try:
            img_file = data_path+"/"+img
            result = inference_detector(model, img_file)
            # test a single image and show the results
            # visualize the results in a new window
            # or save the visualization results to image files
            #filename = outdir +"/" + img.split(".")[0]+"_result.jpg"
            #model.show_result(img, result,score_thr, out_file=filename)
            filename = outdir +"/" + img.split(".")[0]+".json"
            labelmejson = inferTolabelme(result,img_file,score_thr)
            write4json(labelmejson,filename)
        #except :
        #    print("Invalid file : {}".format(img))
