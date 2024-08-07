from torch import nn as nn

from timm.models.layers.helpers import to_2tuple
from timm.models.layers.trace_utils import _assert
from torch.nn.modules.dropout import Dropout

class TransmissionAdapter(nn.Module):

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans , embed_dim // 4 , kernel_size=patch_size, stride=patch_size, bias=bias),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4 , embed_dim // 2 , kernel_size=patch_size, stride=patch_size, bias=bias),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2 , embed_dim , kernel_size=patch_size, stride=patch_size, bias=bias),
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.prompt_dropout = Dropout(0.0)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim , embed_dim) , 
            nn.GELU() ,
            nn.Linear(embed_dim , embed_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        x = self.mlp(x)
        x = self.prompt_dropout(x)
        return x