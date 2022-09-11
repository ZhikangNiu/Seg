# -*- coding: utf-8 -*-
# @Time    : 2022-09-05 19:33
# @Author  : Zhikang Niu
# @FileName: models.py
# @Software: PyCharm
from typing import Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from timm.models.layers import DropPath
from einops import rearrange


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dwconv = DWConv(hidden_dim)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
        k, v = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, mlp_ratio=4, dpr=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: Tensor, H, W) -> Tensor:
        x, attn = self.attn(self.norm1(x), H, W)
        x = x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x, attn


class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=64, patch_size=7, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, patch_size // 2)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


pvtv2_settings = {
    'B1': [2, 2, 2, 2],  # depths
    'B2': [3, 4, 6, 3],
    'B3': [3, 4, 18, 3],
    'B4': [3, 8, 27, 3],
    'B5': [3, 6, 40, 3]
}


class PVTv2(nn.Module):
    def __init__(self, model_name: str = 'B1') -> None:
        super().__init__()
        depths = pvtv2_settings[model_name]
        embed_dims = [64, 128, 320, 512]
        drop_path_rate = 0.1
        self.channels = embed_dims
        # patch_embed
        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # transformer encoder
        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, 8, dpr[cur + i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, 8, dpr[cur + i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, 4, dpr[cur + i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, 4, dpr[cur + i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x, attn1 = blk(x, H, W)
            # print(attn1.shape)
        x1 = self.norm1(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # stage 2
        x, H, W = self.patch_embed2(x1)
        for blk in self.block2:
            x, attn2 = blk(x, H, W)
            # print(attn2.shape)
        x2 = self.norm2(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # stage 3
        x, H, W = self.patch_embed3(x2)
        for blk in self.block3:
            x, attn3 = blk(x, H, W)
            # print(attn3.shape)
        x3 = self.norm3(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # stage 4
        x, H, W = self.patch_embed4(x3)
        for blk in self.block4:
            x, attn4 = blk(x, H, W)
            # print(attn4.shape)
        x4 = self.norm4(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        return (x1, x2, x3, x4), (attn1, attn2, attn3, attn4)


class MLP_Head(nn.Module):
    def __init__(self, dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class PatchEmbed_Head(nn.Module):
    def __init__(self, patch_size=4, in_ch=3, dim=96, type='pool') -> None:
        super().__init__()
        self.patch_size = patch_size
        self.type = type
        self.dim = dim

        if type == 'conv':
            self.proj = nn.Conv2d(in_ch, dim, patch_size, patch_size, groups=patch_size * patch_size)
        else:
            self.proj = nn.ModuleList([
                nn.MaxPool2d(patch_size, patch_size),
                nn.AvgPool2d(patch_size, patch_size)
            ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        if W % self.patch_size != 0:
            x = F.pad(x, (0, self.patch_size - W % self.patch_size))
        if H % self.patch_size != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size - H % self.patch_size))

        if self.type == 'conv':
            x = self.proj(x)
        else:
            x = 0.5 * (self.proj[0](x) + self.proj[1](x))
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).view(-1, self.dim, Wh, Ww)
        return x


class LawinAttn(nn.Module):
    def __init__(self, in_ch=512, head=4, patch_size=8, reduction=2) -> None:
        super().__init__()
        self.head = head

        self.position_mixing = nn.ModuleList([
            nn.Linear(patch_size * patch_size, patch_size * patch_size)
            for _ in range(self.head)])

        self.inter_channels = max(in_ch // reduction, 1)
        self.g = nn.Conv2d(in_ch, self.inter_channels, 1)
        self.theta = nn.Conv2d(in_ch, self.inter_channels, 1)
        self.phi = nn.Conv2d(in_ch, self.inter_channels, 1)
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch)
        )

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        B, C, H, W = context.shape
        context = context.reshape(B, C, -1)
        context_mlp = []

        for i, pm in enumerate(self.position_mixing):
            context_crt = context[:, (C // self.head) * i:(C // self.head) * (i + 1), :]
            context_mlp.append(pm(context_crt))

        context_mlp = torch.cat(context_mlp, dim=1)
        context = context + context_mlp
        context = context.reshape(B, C, H, W)

        g_x = self.g(context).view(B, self.inter_channels, -1)
        g_x = rearrange(g_x, "b (h dim) n -> (b h) dim n", h=self.head)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(query).view(B, self.inter_channels, -1)
        theta_x = rearrange(theta_x, "b (h dim) n -> (b h) dim n", h=self.head)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(context).view(B, self.inter_channels, -1)
        phi_x = rearrange(phi_x, "b (h dim) n -> (b h) dim n", h=self.head)

        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= theta_x.shape[-1] ** 0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)

        y = torch.matmul(pairwise_weight, g_x)
        y = rearrange(y, "(b h) n dim -> b n (h dim)", h=self.head)
        y = y.permute(0, 2, 1).contiguous().reshape(B, self.inter_channels, *query.shape[-2:])

        output = query + self.conv_out(y)
        return output


class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)  # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))


class LawinHead(nn.Module):
    def __init__(self, in_channels: list, embed_dim=512, num_classes=19) -> None:
        super().__init__()
        for i, dim in enumerate(in_channels):
            self.add_module(f"linear_c{i + 1}", MLP_Head(dim, 48 if i == 0 else embed_dim))

        self.lawin_8 = LawinAttn(embed_dim, 64)
        self.lawin_4 = LawinAttn(embed_dim, 16)
        self.lawin_2 = LawinAttn(embed_dim, 4)
        self.ds_8 = PatchEmbed_Head(8, embed_dim, embed_dim)
        self.ds_4 = PatchEmbed_Head(4, embed_dim, embed_dim)
        self.ds_2 = PatchEmbed_Head(2, embed_dim, embed_dim)

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(embed_dim, embed_dim)
        )
        self.linear_fuse = ConvModule(embed_dim * 3, embed_dim)
        self.short_path = ConvModule(embed_dim, embed_dim)
        self.cat = ConvModule(embed_dim * 5, embed_dim)

        self.low_level_fuse = ConvModule(embed_dim + 48, embed_dim)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def get_lawin_att_feats(self, x: Tensor, patch_size: int):
        _, _, H, W = x.shape
        query = F.unfold(x, patch_size, stride=patch_size)
        query = rearrange(
            query,
            'b (c ph pw) (nh nw) -> (b nh nw) c ph pw',
            ph=patch_size, pw=patch_size,
            nh=H // patch_size, nw=W // patch_size
        )
        outs = []

        for r in [8, 4, 2]:
            context = F.unfold(x, patch_size * r, stride=patch_size, padding=int((r - 1) / 2 * patch_size))
            # print(context.shape)
            context = rearrange(
                context,
                "b (c ph pw) (nh nw) -> (b nh nw) c ph pw",
                ph=patch_size * r, pw=patch_size * r,
                nh=H // patch_size, nw=W // patch_size
            )
            context = getattr(self, f"ds_{r}")(context)
            output = getattr(self, f"lawin_{r}")(query, context)
            output = rearrange(output, "(b nh nw) c ph pw -> b c (nh ph) (nw pw)", ph=patch_size, pw=patch_size,
                               nh=H // patch_size, nw=W // patch_size)
            outs.append(output)
        return outs

    def forward(self, features):
        B, _, H, W = features[1].shape
        outs = [self.linear_c2(features[1]).permute(0, 2, 1).reshape(B, -1, *features[1].shape[-2:])]

        for i, feature in enumerate(features[2:]):
            cf = eval(f"self.linear_c{i + 3}")(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))

        feat = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        B, _, H, W = feat.shape

        ## Lawin attention spatial pyramid pooling
        feat_short = self.short_path(feat)
        feat_pool = F.interpolate(self.image_pool(feat), size=(H, W), mode='bilinear', align_corners=False)
        feat_lawin = self.get_lawin_att_feats(feat, 8)
        output = self.cat(torch.cat([feat_short, feat_pool, *feat_lawin], dim=1))

        ## Low-level feature enhancement
        c1 = self.linear_c1(features[0]).permute(0, 2, 1).reshape(B, -1, *features[0].shape[-2:])
        output = F.interpolate(output, size=features[0].shape[-2:], mode='bilinear', align_corners=False)
        fused = self.low_level_fuse(torch.cat([output, c1], dim=1))

        seg = self.linear_pred(self.dropout(fused))
        return seg


class PVTv2_Lawin(nn.Module):
    def __init__(self, PVT_model_name: str = 'B1', num_classes: int = 19,pretrained=False):
        super().__init__()
        self.backbone = PVTv2(PVT_model_name)
        # if pretrained:
        #     state_dict = self.backbone.state_dict()
        #     pretrain_backbone = torch.load('pvt_v2_b1.pth')
        #     new_state_dict = {"backbone." + k: v for k, v in pretrain_backbone.items() if "backbone." + k in state_dict}
        #     print(f'load PVTv2 pretrained weights')
        #     self.backbone.load_state_dict(new_state_dict, strict=False)
        # self.backbone = self.backbone.load_state_dict(torch.load('./pvt_v2_b1.pth'))
        self.head =  LawinHead([64, 128, 320, 512], num_classes=num_classes)

    def forward(self, x):
        out_feature, attn = self.backbone(x)
        out = self.head(out_feature)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out, attn


if __name__ == '__main__':
    model = PVTv2_Lawin('B4', num_classes=9)
    # from torchinfo import summary
    # summary(model, (2, 3, 512, 512))
    # state_dict = model.state_dict()
    #model.load_state_dict(torch.load('pvt_v2_b1.pth'), strict=False)
    # pth_state = torch.load('pvt_v2_b1.pth')
    # for k, v in pth_state.items():
    #     print(k)
    # new_state_dict = {"backbone."+k:v for k,v in pth_state.items() if "backbone."+k in state_dict}
    # print(new_state_dict.keys())
    # model.load_state_dict(new_state_dict, strict=False)
    x = torch.zeros(2, 3, 512, 512)
    outs = model(x)[0]
    for y in outs:
        print(y.shape)
