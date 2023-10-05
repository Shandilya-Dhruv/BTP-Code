from math import sqrt
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
from skimage import measure
import os
from model import *

from models.attention import *
from models.backbone import *
from models.transformer import Transformer

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        self.mode = mode
        if model_name == 'DNANet':
            if mode == 'train':
                self.model = DNANet(mode='train')
            else:
                self.model = DNANet(mode='test')  
        elif model_name == 'ACM':
            self.model = ACM()
        elif model_name == 'ALCNet':
            self.model = ALCNet()
        elif model_name == 'IAANet':
            #  Model #
            #Region Proposal Networks
            region_module = torch.load('model/IAANet/pretrained/rpn.pt', map_location=torch.device('cpu'))
            region_module.trainable = False

            #Attention Encoder
            attention_module = Transformer(num_encoder_layers=4, d_model=512)
            #IAANet
            Model = attention(attention_module, region_module, pos='cosin', d_model=512)
            self.model = Model

        
        self.cal_loss = FocalLoss()
        
    def forward(self, img):
        pred = self.model(img)
        _, _, h, w = img.shape
        if self.model_name=='IAANet':
            # output from -> (detect_output, seg_output, mask_maps, target_boxes)
            # Not [2] but the dimensions match -> mask_maps is boolean tensor
            # take seg_output and mask_maps and combine to give following result
            if self.mode=='train':
                _, output, mask_maps, _ = pred
            else:
                output, mask_maps, _, _ = pred
            probability_map = torch.zeros((h, w), dtype=torch.float)
            if output is not None:
                output = output.squeeze()
                output = output.sigmoid()
                mask_maps = mask_maps.squeeze()

                probability_map[~mask_maps] = output
            shape = probability_map.shape
            probability_map = probability_map.reshape(1, 1, *shape)
            # probability_map = torch.from_numpy(probability_map)
            return probability_map
        return pred

    def loss(self, pred, gt_mask):
        target_mask, avg_factor = gt_mask, max(1, (gt_mask.eq(1)).sum())
        loss = self.cal_loss(pred, target_mask, avg_factor=avg_factor)
        return loss
        
    def update_gt(self, pred, gt_masks, thresh_Tb, thresh_k, size):
        bs, c, feat_h, feat_w = pred.shape
        update_gt_masks = gt_masks.clone()
        background_length = 33
        target_length = 3
        
        label_image = measure.label((gt_masks[0,0,:,:]>0.5).cpu())
        for region in measure.regionprops(label_image, cache=False):
            cur_point_mask = pred.new_zeros(bs, c, feat_h, feat_w)
            cur_point_mask[0, 0, int(region.centroid[0]), int(region.centroid[1])] = 1
            nbr_mask = ((F.conv2d(cur_point_mask, weight=torch.ones(1,1,background_length,background_length).to(gt_masks.device), stride=1, padding=background_length//2))>0).float()
            targets_mask = ((F.conv2d(cur_point_mask, weight=torch.ones(1,1,target_length,target_length).to(gt_masks.device), stride=1, padding=target_length//2))>0).float()
            
            ### Candidate Pixels Extraction
            max_limitation = size[0] * size[1] * 0.0015
            threshold_start = (pred * nbr_mask ).max()*thresh_Tb
            threshold_delta = thresh_k * ((pred * nbr_mask ).max() - threshold_start) * (len(region.coords)/max_limitation).to(gt_masks.device)
            threshold = threshold_start + threshold_delta
            thresh_mask = (pred * nbr_mask > threshold).float()
            ### False Alarm Elimination
            label_image = measure.label((thresh_mask[0,:,:,:]>0).cpu())
            
            if label_image.max() > 1:
                for num_cur in range(label_image.max()):
                    curr_mask = thresh_mask * torch.tensor(label_image == (num_cur + 1)).float().unsqueeze(0).to(gt_masks.device)         
                    if (curr_mask * targets_mask).sum() == 0:
                        thresh_mask = thresh_mask - curr_mask
                        
            
            ### Average Weighted Summation
            target_patch = (update_gt_masks * thresh_mask + pred * thresh_mask)/2
            background_patch = update_gt_masks * (1-thresh_mask)
            update_gt_masks = background_patch + target_patch
            
        
        ### Ensure initial GT point label    
        update_gt_masks = torch.max(update_gt_masks, (gt_masks==1).float())
        return update_gt_masks