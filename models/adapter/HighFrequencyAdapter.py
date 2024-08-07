import torch
from timm.models.layers.helpers import to_2tuple
from torch import nn as nn

class HighFrequencyAdapter(torch.nn.Module):

    def __init__(self):
        super(HighFrequencyAdapter, self).__init__(
            output_embedding_dims = [8, 16, 32, 64]
        )
        self.embed_dim = 16
        self.patch_embed1 = Adapter_PatchEmbed(img_size=384 ,patch_size=4 , in_chans=3 , embed_dim=self.embed_dim)
        self.patch_embed2 = Adapter_PatchEmbed(img_size=384 ,patch_size=8 , in_chans=3 , embed_dim=2 * self.embed_dim)
        self.patch_embed3 = Adapter_PatchEmbed(img_size=384 ,patch_size=16 , in_chans=3 , embed_dim=4 * self.embed_dim)
        self.patch_embed4 = Adapter_PatchEmbed(img_size=384 ,patch_size=32 , in_chans=3 , embed_dim=8 * self.embed_dim)
        self.lightweight_mlp1 = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),nn.GELU())
        self.lightweight_mlp2 = nn.Sequential(nn.Linear(2 * self.embed_dim, 2 * self.embed_dim),nn.GELU())
        self.lightweight_mlp3 = nn.Sequential(nn.Linear(4 * self.embed_dim, 4 * self.embed_dim),nn.GELU())
        self.lightweight_mlp4 = nn.Sequential(nn.Linear(8 * self.embed_dim, 8 * self.embed_dim),nn.GELU())
        self.mlp1 = nn.Linear(self.embed_dim , output_embedding_dims[0] * self.embed_dim)
        self.mlp2 = nn.Linear(2 * self.embed_dim , output_embedding_dims[1] * self.embed_dim)
        self.mlp3 = nn.Linear(4 * self.embed_dim , output_embedding_dims[2] * self.embed_dim)
        self.mlp4 = nn.Linear(8 * self.embed_dim , output_embedding_dims[3] * self.embed_dim)
        
    def fft(self, x, rate):
        mask = torch.zeros(x.shape).to(x.device)
        w, h = x.shape[-2:]
        line = int((w * h * rate) ** .5 // 2)
        mask[:, :, w//2-line:w//2+line, h//2-line:h//2+line] = 1
        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
        fft = fft * (1 - mask)
        fr = fft.real
        fi = fft.imag
        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real
        inv = torch.abs(inv)
        return inv

    def forward(self, x):
        x = self.fft(x , 0.25)
        x1_save = self.patch_embed1(x)
        x1 = x1_save.permute(0 , 2 , 3 , 1)
        x1 = self.lightweight_mlp1(x1)
        x1 = self.mlp1(x1)
        x1 = x1.permute(0 , 3 , 1 , 2)
        x2 = self.patch_embed2(x)
        x2 = x2.permute(0 , 2 , 3 , 1)
        x2 = self.lightweight_mlp2(x2)
        x2 = self.mlp2(x2)
        x2 = x2.permute(0 , 3 , 1 , 2)
        x3 = self.patch_embed3(x)
        x3 = x3.permute(0 , 2 , 3 , 1)
        x3 = self.lightweight_mlp3(x3)
        x3 = self.mlp3(x3)
        x3 = x3.permute(0 , 3 , 1 , 2)
        x4 = self.patch_embed4(x)
        x4 = x4.permute(0 , 2 , 3 , 1)
        x4 = self.lightweight_mlp4(x4)
        x4 = self.mlp4(x4)
        x4 = x4.permute(0 , 3 , 1 , 2)

        return x1 , x2 , x3 , x4

class Adapter_PatchEmbed(nn.Module):

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x