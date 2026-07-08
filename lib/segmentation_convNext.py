import torch
import torch.nn as nn

from lib.backbone_ConvNext import MultiModel_ConvNeXt
from .mask_predictor import SimpleDecoding
from .backbone import MultiModalSwinTransformer
from ._utils import LAVT, LAVTOne


__all__ = ['lavt', 'lavt_one']


# LAVT
def _segm_lavt(pretrained, args):
    # initialize the SwinTransformer backbone with the specified version
    if args.swin_type == 'tiny':
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'small':
        embed_dim = 96
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'base':
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
    elif args.swin_type == 'large':
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
    else:
        assert False
    # args.window12 added for test.py because state_dict is loaded after model initialization
    if 'window12' in pretrained or args.window12:
        print('Window size 12!')
        window_size = 12
    else:
        window_size = 7

    if args.mha:
        mha = args.mha.split('-')  # if non-empty, then ['a', 'b', 'c', 'd']
        mha = [int(a) for a in mha]
    else:
        mha = [1, 1, 1, 1]

    out_indices = (0, 1, 2, 3)
    backbone = MultiModalSwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                         window_size=window_size,
                                         ape=False, drop_path_rate=0.3, patch_norm=True,
                                         out_indices=out_indices,
                                         use_checkpoint=False, num_heads_fusion=mha,
                                         fusion_drop=args.fusion_drop
                                         )
    if pretrained:
        print('Initializing Multi-modal Swin Transformer weights from ' + pretrained)
        backbone.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize Multi-modal Swin Transformer weights.')
        backbone.init_weights()

    model_map = [SimpleDecoding, LAVT]

    classifier = model_map[0](8*embed_dim)
    base_model = model_map[1]

    model = base_model(backbone, classifier)
    return model


def _load_model_lavt(pretrained, args):
    model = _segm_lavt(pretrained, args)
    return model


def lavt(pretrained='', args=None):
    return _load_model_lavt(pretrained, args)


###############################################
# LAVT One: put BERT inside the overall model #
###############################################
def _segm_lavt_one(pretrained, args):
    # initialize the SwinTransformer backbone with the specified version
    if args.swin_type == 'tiny':
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'small':
        embed_dim = 96
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'base':
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
    elif args.swin_type == 'large':
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
    else:
        assert False
    # args.window12 added for test.py because state_dict is loaded after model initialization
    if 'window12' in pretrained or args.window12:
        print('Window size 12!')
        window_size = 12
    else:
        window_size = 7

    if args.mha:
        mha = args.mha.split('-')  # if non-empty, then ['a', 'b', 'c', 'd']
        mha = [int(a) for a in mha]
    else:
        mha = [1, 1, 1, 1]

    out_indices = (0, 1, 2, 3)
    backbone = MultiModalSwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                         window_size=window_size,
                                         ape=False, drop_path_rate=0.3, patch_norm=True,
                                         out_indices=out_indices,
                                         use_checkpoint=False, num_heads_fusion=mha,
                                         fusion_drop=args.fusion_drop,
                                         # frozen_stages=args.frozen_stages,
                                         # only_fusion=args.only_fusion,
                                         )
    if pretrained:
        print('Initializing Multi-modal Swin Transformer weights from ' + pretrained)
        backbone.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize Multi-modal Swin Transformer weights.')
        backbone.init_weights()

    model_map = [SimpleDecoding, LAVTOne]
    classifier = model_map[0](8*embed_dim)
    base_model = model_map[1]

    model = base_model(backbone, classifier, args)
    return model

def _segm_lavt_one_ConvNext(pretrained, args):
    # initialize the SwinTransformer backbone with the specified version
    if args.conv_type == 'tiny':
        embed_dim = 96
        backbone = MultiModel_ConvNeXt(embed_dim,depths=[3, 3, 9, 3],
                                    dims=[96, 192, 384, 768],
                                    drop_path_rate=0.2)
    elif args.conv_type == 'small':
        embed_dim = 96
        backbone = MultiModel_ConvNeXt(embed_dim,depths=[3, 3, 27, 3],
                                    dims=[96, 192, 384, 768],
                                    )
    elif args.conv_type == 'base':
        embed_dim = 128
        backbone = MultiModel_ConvNeXt(embed_dim,depths=[3, 3, 27, 3],
                                    dims=[128, 256, 512, 1024],
                                    )
    elif args.conv_type == 'large':
        embed_dim = 192
        backbone = MultiModel_ConvNeXt(embed_dim,depths=[3, 3, 27, 3],
                                    dims=[192, 384, 768, 1536],
                                    )
    elif args.conv_type == 'xlarge':
        embed_dim = 192
        backbone = MultiModel_ConvNeXt(embed_dim,depths=[3, 3, 27, 3],
                                    dims=[192, 384, 768, 1536],
                                    )
    else:
        assert False


    # backbone = MultiModel_ConvNeXt(embed_dim)
    if pretrained:
        print('Initializing Multi-modal ConvNext weights from ' + pretrained)
        backbone.init_weights_conv(pretrained=pretrained)
    else:
        print('Randomly initialize Multi-modal ConvNext weights.')
        backbone.init_weights_conv()

    model_map = [SimpleDecoding, LAVTOne]
    classifier = model_map[0](8*embed_dim)
    base_model = model_map[1]

    model = base_model(backbone, classifier, args)
    return model
def _load_model_lavt_one(pretrained, args):
    if args.backbone_type == 'ConvNext':
        model=_segm_lavt_one_ConvNext(pretrained, args)
    else:
        model = _segm_lavt_one(pretrained, args)
    return model


def lavt_one(pretrained='', args=None):
    return _load_model_lavt_one(pretrained, args)
if __name__ == '__main__':
    from torchsummary import summary

    embed_dim = 128
    depths = [2, 2, 18, 2]
    num_heads = [4, 8, 16, 32]
    window_size = 12
    mha = [1, 1, 1, 1]
    out_indices = (0, 1, 2, 3)

    backbone = MultiModalSwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                         window_size=window_size,
                                         ape=False, drop_path_rate=0.3, patch_norm=True,
                                         out_indices=out_indices,
                                         use_checkpoint=False, num_heads_fusion=mha,
                                         fusion_drop=0.0,
                                         # frozen_stages=args.frozen_stages,
                                         # only_fusion=args.only_fusion,
                                         )
    backbone.init_weights()
    model=backbone
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    print(model)
    x = torch.randn(1,3,480,480,device=device)
    l=torch.randn(1,768,20,device=device)
    l_mask=torch.randn(1,20,1,device=device)
    y = model(x,l,l_mask)
    print(y)

    summary(model, input_size=(3, 480, 480))
    from thop import profile
    input = torch.randn(1, 3, 480, 480)
    flops, params = profile(model, inputs=(input,))
    print("flops:{:.3f}G".format(flops/1e9))
    print("params:{:.3f}M".format(params/1e6))