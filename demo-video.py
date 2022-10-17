import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

import torch, gc

gc.collect()
torch.cuda.empty_cache()


DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    i=0
    with torch.no_grad():
        capture = cv2.VideoCapture(args.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        ret,image1 = capture.read()
        print(image1.shape)
        width = int(image1.shape[1])
        height = int(image1.shape[0])
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image1 = image1[None].to(DEVICE)
        #width = int(img.shape[1])*2
        out = cv2.VideoWriter(args.save_path,fourcc,30,(width,height*2))
        if capture.isOpened():
            while True:
                ret,image2 = capture.read()
                if not ret:break
                image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
                image2 = image2[None].to(DEVICE)
                pre = image2
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                image1 = image1[0].permute(1,2,0).cpu().numpy()
                flow_up = flow_up[0].permute(1,2,0).cpu().numpy()
                # map flow to rgb image
                flow_up = flow_viz.flow_to_image(flow_up)
                img_flo = np.concatenate([image1, flow_up], axis=0)
                img_flo = img_flo[:, :, [2,1,0]]
                out.write(np.uint8(img_flo))
                image1 = pre
        else:
            print("open video error!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/raft-things.pth',help="restore checkpoint")
    parser.add_argument('--video_path', default='E:/bilibiliMP4/huaban3.mp4',help="video data")
    parser.add_argument('--save_path', default='E:/raft-video/res_1.mp4',help="result video save path")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--device', default='0',help='assign device')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
    demo(args)

