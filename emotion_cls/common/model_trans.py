from common.temporal_convolution import TemporalModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from functools import partial
from einops import rearrange

from timm.models.layers import DropPath


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformNet(nn.Module):
    def __init__(
        self,
        kp=34,
        dim=3,
        hidden_dim=256
    ):
        super().__init__()
        
        # layers
        self.norm1 = nn.BatchNorm2d(64)
        self.norm2 = nn.BatchNorm2d(128)
        self.norm3 = nn.BatchNorm2d(hidden_dim)
        self.norm4 = nn.BatchNorm1d(hidden_dim//2)
        self.norm5 = nn.BatchNorm1d(hidden_dim//4)
        
        self.cnn1 = nn.Conv2d(1,  64, (1,dim))
        self.cnn2 = nn.Conv2d(64, 128, (1,1))
        self.cnn3 = nn.Conv2d(128, hidden_dim, (1,1))
        
        self.maxpool = nn.MaxPool2d((kp,1))
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.fc3 = nn.Linear(hidden_dim//4, dim*dim)
        
        # activation functions
        self.act_layer = nn.ReLU()
        
        # params
        self.K = dim
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        # x.shape = (b*f, kp, 3)
        batch_size = x.shape[0]
        x = torch.unsqueeze(x, -1)
        x = x.permute((0,3,1,2)) # x.shape = (b*f, 1, kp, 3)
        x = self.act_layer(self.norm1(self.cnn1(x)))
        x = self.act_layer(self.norm2(self.cnn2(x)))
        x = self.act_layer(self.norm3(self.cnn3(x)))
        
        # x.shape = (b*f, 128, kp, 1)
        x = self.maxpool(x)
        x = x.view(-1, self.hidden_dim)
        x = self.act_layer(self.norm4(self.fc1(x)))
        x = self.act_layer(self.norm5(self.fc2(x)))
        x = self.fc3(x)
        
        B = Variable(torch.eye(self.K).flatten()).view(1,self.K*self.K).repeat(batch_size,1)
        if x.is_cuda:
            B = B.cuda()
        x = x + B
        return x.view(-1, self.K, self.K)

class EmotionClassifier(nn.Module):
    def __init__(
        self,
        kp=34,
        feature_dim=3,
        hidden_dim=256,
        channels=1024,
        out_dim=64,
        cls=7,
        trans=False,
    ):
        super().__init__()
        
        # layer
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        if trans:
            self.feature_transform = TransformNet(kp, 3, hidden_dim)
        self.feature_spatial = nn.Linear(3, feature_dim)
        
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, kp, feature_dim))
        self.pos_drop = nn.Dropout(p=0.25)
        dpr = [x.item() for x in torch.linspace(0, 0.1, 3)]  # stochastic depth decay rule
        self.spatial_blocks = nn.ModuleList([
            Block(
                dim=feature_dim, num_heads=4, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(3)])
        self.spatial_norm = norm_layer(feature_dim)
        
        embed_dim = kp*feature_dim
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, 27, embed_dim))
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=4, mlp_ratio=2, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(3)])
        self.temporal_norm = norm_layer(embed_dim)
        self.weighted_mean = torch.nn.Conv1d(in_channels=27, out_channels=1, kernel_size=1)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.Linear(256, out_dim)
        )
        
        # self.temporal_convolution = TemporalModel(kp, feature_dim, out_dim, [3,3,3], channels=channels)
        
        self.final = nn.Linear(out_dim, cls)
        
        # activation functions
        self.act_layer = nn.ReLU()
        
        # param
        self.trans = trans
    
    def spatial_forward_features(self, x):
        b, f, p, c = x.shape
        x = rearrange(x, 'b f p c -> (b f) p c', )
        
        x += self.spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.spatial_blocks:
            x = blk(x)

        x = self.spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x
    
    def forward_features(self, x):
        b  = x.shape[0]
        x += self.temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.temporal_norm(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        x = self.weighted_mean(x)
        x = x.view(b, -1)
        return x

    def forward(self, x, y):
        # x.shape = (batch, frames, keypoints, 3), keypoints=34
        b, f, p, _ = x.shape
        
        if self.trans:
            x = rearrange(x, 'b f p c -> (b f) p c', )
            T_input = self.feature_transform(x) # (b*f, p, feature_dim)
            x = torch.bmm(x, T_input)
            x = x.view(b, f, p, -1)
        
        x = self.feature_spatial(x)
        x = self.spatial_forward_features(x)
        
        x = self.forward_features(x)
        x = self.head(x)
        x = self.final(x)
        
        # temporal convolution
        # x = self.temporal_convolution(x)
        # x = self.final(self.act_layer(x))
        # x = x.view(b, -1)
        
        y = torch.squeeze(y)
        return x.max(1)[1], F.cross_entropy(x, y.long())

if __name__ == '__main__':
    test_layer = EmotionClassifier()
    
    model_params = 0
    for parameter in test_layer.parameters():
        model_params += parameter.numel()
    print('Trainable parameter count:', model_params)
    
    A = torch.randn(8,27,34,3)
    B = torch.randn(8,1)
    test_layer(A)