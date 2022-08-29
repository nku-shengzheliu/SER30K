# Code based on the Pyramid Vision Transformer
# https://github.com/whai362/PVT
# Licensed under the Apache License, Version 2.0

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_

from pvt_local_final import PyramidVisionTransformer
# from pvt_local_laststage import PyramidVisionTransformer
# from pvt_local_2 import PyramidVisionTransformer
# from pvt import PyramidVisionTransformer
from block import FuseBlock
from pytorch_pretrained_bert.modeling import BertModel

class LORA(nn.Module):
    def __init__(self, args):
        super(LORA, self).__init__()
        num_classes=args.nb_classes
        self.tunebert = args.tunebert
        self.bert_dim = args.bert_dim
        self.hidden_dim = args.hidden_dim
        ## Vision model--PVT small: https://github.com/whai362/PVT
        self.vision_model = PyramidVisionTransformer(
                                patch_size=4, 
                                embed_dims=[64, 128, 320, 512], 
                                num_heads=[1, 2, 5, 8], 
                                mlp_ratios=[8, 8, 4, 4], 
                                qkv_bias=True,
                                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                depths=[3, 4, 6, 3], 
                                sr_ratios=[8, 4, 2, 1], 
                                num_classes=args.nb_classes,
                                drop_rate=args.drop,
                                drop_path_rate=args.drop_path,
                                alpha = args.alpha,
                                locals= args.locals)
        ## Text model
        self.text_model = BertModel.from_pretrained(args.bert_model)
        ## Visual-Language Fusion model
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        token_nums = [64, 32, 20, 8]
        local_num = 8  # must have last stage
        for i in range(4):
            if args.locals[i]:
                local_num = local_num+token_nums[i]
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+sum(args.locals)+local_num*args.alpha+30, self.hidden_dim))  
        # alpha=8, locals:
        # [1, 1, 1, 0]: [1, 996, 512]
        # [0, 1, 1, 0]: [1, 483, 512]
        # [0, 0, 1, 0]: [1, 226, 512]
        # [0, 0, 0, 0]: [1,  65, 512]
        self.v_proj = nn.Linear(self.vision_model.embed_dims[3], self.hidden_dim)
        self.l_proj = nn.Linear(self.bert_dim, self.hidden_dim)
        self.fusion_model = FuseBlock(args)
        ## Prediction Head
        self.head = nn.Linear(self.hidden_dim, num_classes) if num_classes > 0 else nn.Identity()

        self._init_weights()
        
        
    def _init_weights(self):
        ## Fusion
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.normal_(self.v_proj.bias, std=1e-6)
        nn.init.xavier_uniform_(self.l_proj.weight)
        nn.init.normal_(self.l_proj.bias, std=1e-6)
        trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        # Head
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.normal_(self.head.bias, std=1e-6)
    
    
    def forward(self, image, word_id, word_mask):
        B = image.shape[0]
        ## Vision features
        fv = self.vision_model(image)
        ## Language features
        all_encoder_layers = self.text_model(word_id, \
            token_type_ids=None, attention_mask=word_mask)
        fl = (all_encoder_layers[-1] + all_encoder_layers[-2]\
             + all_encoder_layers[-3] + all_encoder_layers[-4])/4  # [bs, 30, 768]
        if not self.tunebert:
            fl = fl.detach()
        ## Fusion
        # x = torch.cat([self.cls_token.expand([B, -1, -1]), self.v_proj(fv), self.l_proj(fl)], dim=1)
        x = torch.cat([self.v_proj(fv), self.l_proj(fl)], dim=1)
        x = x+self.pos_embed
        x = self.fusion_model(x)
        ## Prediction
        x = self.head(x)
        return x

if __name__=='__main__':
    parser = argparse.ArgumentParser('LORA training and evaluation script', add_help=False)
    parser.add_argument('--fp32-resume', action='store_true', default=False)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--config', default='configs/pvt/pvt_small.py', type=str, help='config')

    # Vision Model parameters
    parser.add_argument('--model', default='pvt_small', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=448, type=int, help='images input size')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    # Language Model parameters
    parser.add_argument('--tunebert', default=False, type=bool)
    parser.add_argument('--bert-dim', default=768, type=int)
    parser.add_argument('--bert-model', default='bert-base-uncased', type=str)
    # Fusion Model parameters
    parser.add_argument('--hidden-dim', default=768, type=int)
    parser.add_argument('--fuse-mlp-dim', default=3072, type=int)
    parser.add_argument('--fuse-dropout-rate', default=0.1, type=float)
    parser.add_argument('--fuse-num-heads', default=12, type=int)
    parser.add_argument('--fuse-attention-dropout-rate', default=0.0, type=float)
    
    parser.add_argument('--alpha', default=8, type=int, help='alpha')
    parser.add_argument('--locals', default=[1, 1, 1, 0], type=list, help='locals')
    
    args = parser.parse_args()
    args.nb_classes = 7  # SER 7 classes
    model = LORA(args).cuda()
    ## Load checkpoint
    checkpoint = torch.load('weights/pvt_small.pth', map_location='cpu')
    if 'model' in checkpoint:
        checkpoint_model = checkpoint['model']
    else:
        checkpoint_model = checkpoint
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model:
            del checkpoint_model[k]
    model.vision_model.load_state_dict(checkpoint_model, strict=False)
    
    x = torch.randn([1, 3, 448, 448]).cuda()
    y = model(x, 0, 0)
    print(y.shape)