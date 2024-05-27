import os
import cv2 
import torch 

from DPIR.models.network_unet import UNetRes as net

def load_img(img_path, img_size=256):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.blur(img, (5, 5))
    return img

def load_model(model_name, model_pool):
    model_path = os.path.join(model_pool, model_name+'.pth')
    n_channels = 3
    model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path), strict=True)
    for k, v in model.named_parameters():
            v.requires_grad = False

    return model