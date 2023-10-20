from model.IAANet.models.attention import *
from model.IAANet.models.backbone import *
from model.IAANet.models.transformer import Transformer
# from model.IAANet.utils.datasets import *
# from utils.loss import DetectLoss, SegLoss, ComputerLoss
# from test import test


class IAANet(nn.Module):
    def __init__(self, num_encoder_layers=4, d_model=512, pos="cosin"):
        super(IAANet, self).__init__()
        #Region Proposal Networks
        backbones = backbone()
        region_module = region_propose(backbones)
        # if rpn_pretrained:
        #     region_module = torch.load(rpn_pretrained)
            
        #Attention Encoder
        attention_module = Transformer(d_model, num_encoder_layers)
        #IAANet
        self.Model = attention(attention_module, region_module, pos, d_model)
        self.named_parameters = self.Model.named_parameters()
    
    def forward(self, input):
        return self.Model(input)