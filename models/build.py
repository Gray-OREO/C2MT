# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .Transformer_sriqa import Transformer_iqa


def build_model(config):
    n_cls = config['n_distortions']
    model = Transformer_iqa(img_size=56,
                            patch_size=1,  # ori
                            in_chans=3,
                            num_classes=1,
                            num_dis=n_cls,
                            embed_dim=64,
                            depths=[2, 4],
                            num_heads=[4, 4],
                            window_size=8,
                            mlp_ratio=2.,
                            qkv_bias=True,
                            qk_scale=True,
                            drop_rate=0.0,
                            drop_path_rate=0.1,
                            ape=True,
                            patch_norm=True,
                            use_checkpoint=False)

    return model
