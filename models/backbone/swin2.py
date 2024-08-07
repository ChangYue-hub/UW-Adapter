import timm

from .swin_common import _make_swin_backbone
from .swin_transformer_v2 import SwinTransformerV2

def _make_pretrained_swin2l24_384(pretrained, hooks=None):
    model = SwinTransformerV2(
        img_size= 384,
        window_size= 24,
        embed_dim= 192,
        depths= (2 , 2 , 18 , 2),
        num_heads= (6 , 12 , 24 , 48),
        pretrained_window_sizes= (12 , 12 , 12 , 6)
    )
    hooks = [1, 1, 17, 1] if hooks == None else hooks
    return _make_swin_backbone(
        model,
        hooks=hooks
    )


def _make_pretrained_swin2b24_384(pretrained, hooks=None):
    model = SwinTransformerV2(
        img_size= 384,
        window_size= 24,
        embed_dim= 128,
        depths= (2 , 2 , 18 , 2),
        num_heads= (4 , 8 , 16 , 32),
        pretrained_window_sizes= (12 , 12 , 12 , 6)
    )
    hooks = [1, 1, 17, 1] if hooks == None else hooks
    return _make_swin_backbone(
        model,
        hooks=hooks
    )
