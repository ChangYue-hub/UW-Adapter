from __future__ import absolute_import, division, print_function

import os
import glob
import argparse
import numpy as np
import PIL.Image as pil
import torch
from torchvision.transforms import Compose
import torch.nn as nn
import cv2
from .utils.util import read_image, write_depth
from models.adapter.HighFrequencyAdapter import HighFrequencyAdapter
from models.adapter.TransmissionAdapter import TransmissionAdapter
from models.others.JTnet import JTnet
from models.backbone.dpt_depth import DPTDepthModel
from models.backbone.transforms import Resize, NormalizeImage, PrepareForNet

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for UW-Adapter models.')
    parser.add_argument('--image_path' ,
                        type=str,
                        help='path to a folder of images', required=True)
    parser.add_argument('--output_path' ,
                        type=str,
                        help='path to a folder of output', required=True)
    parser.add_argument('--JTNet_path' ,
                        type=str,
                        help='path to the JTNet', required=True)
    parser.add_argument('--backbone_path' ,
                        type=str,
                        help='path to the backbone', required=True)
    parser.add_argument('--t_adapter_path' ,
                        type=str,
                        help='path to the T-Adapter', required=True)
    parser.add_argument('--h_adapter_path' ,
                        type=str,
                        help='path to the H-Adapter', required=True)
    parser.add_argument('--backbone' ,
                        type=str,
                        help='the backbone selected to use', required=True)
    return parser.parse_args()

def process(device, model,  model_JT , model_t_adapter , model_h_adapter , image, target_size):
    global first_execution
    sample = torch.from_numpy(image).to(device).unsqueeze(0)
    HFComponent  = model_h_adapter(sample)
    out_J , out_T = model_JT(sample)
    prompt = model_t_adapter(out_T)
    prediction = model.forward(sample , prompt , HFComponent)
    prediction = ( 
        torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=target_size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
    )
    
    return prediction

def test_simple(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(" loading weitghs of backbone from {}".format(args.backbone_path))
    parameters = torch.load(args.backbone_path)
    model_backbone = DPTDepthModel(backbone="swin2b24_384" , non_negative=True)
    model_backbone.load_state_dict(parameters)
    model_backbone.eval()
    model_backbone.to(device)
    
    model_JT = JTnet()
    model_JT.to(device)
    model_JT.eval()
    model_JT.load_state_dict(torch.load(args.JTNet_path))

    embedding_dims_adapters = {
        "swin2b24_384":[8, 16, 32, 64, 128],
        "vitb_rn50_384":[16, 32, 38, 48, 768],
        "vitl16_384":[16, 32, 64, 64, 1024]
    }
    output_dims_t_adapter = embedding_dims_adapters[args.backbone]
    output_dims_h_adapter = []
    for i in range(4):
        output_dims_h_adapter.append(embedding_dims_adapters[args.backbone][i])

    model_t_adapter = TransmissionAdapter(img_size= 384, patch_size= 4, in_chans= 3, embed_dim= output_dims_t_adapter, norm_layer= nn.LayerNorm)
    model_t_adapter.load_state_dict(torch.load(args.t_adapter_path))
    model_t_adapter.eval()
    model_t_adapter.to(device)

    model_h_adapter = HighFrequencyAdapter(output_dims_h_adapter)
    model_h_adapter.load_state_dict(torch.load(args.h_adapter_path))
    model_h_adapter.eval()
    model_h_adapter.to(device)

    net_w, net_h = 384, 384
    keep_aspect_ratio = False
    resize_mode = "minimal"
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = Compose(
    [
        Resize(
            net_w,
            net_h,
            resize_target=None,
            keep_aspect_ratio=keep_aspect_ratio,
            ensure_multiple_of=32,
            resize_method=resize_mode,
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        normalization,
        PrepareForNet(),
    ]
)
    image_names = glob.glob(os.path.join(args.image_path, '*'))
    for idx, image_name in enumerate(image_names):
        image_path = os.path.join(args.image_path , image_name)
        original_image_rgb = read_image(image_path)
        image = transform({"image": original_image_rgb})["image"]
        with torch.no_grad():
            prediction = process(device, model_backbone , model_JT , model_t_adapter , model_h_adapter , image, original_image_rgb.shape[1::-1])
        if args.output_path is not None:
            filename = os.path.join(
                args.output_path, os.path.splitext(os.path.basename(image_name))[0])
            write_depth(filename, prediction, False, bits=1)

if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
