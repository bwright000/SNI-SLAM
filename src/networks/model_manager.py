# This file is a part of SNI-SLAM.

from src.networks.dinov2_seg import DINO2SEG
from src.networks.mlp import head, encoder, feature_fusion
import torch

class ModelManager:
    def __init__(self, cfg):
        self.dim = cfg['model']['c_dim']
        self.hidden_dim = cfg['model']['hidden_dim']

        self.encoder_multires = cfg['model']['encoder']['multires']

        self.pretrained_model_path = cfg['model']['cnn']['pretrained_model_path']
        self.n_classes = cfg['model']['cnn']['n_classes']

        self.img_h = cfg['cam']['H']
        self.img_w = cfg['cam']['W']

        self.crop_edge = cfg['cam']['crop_edge']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = self.get_encoder().cuda()
        self.head = self.get_head().cuda()
        self.cnn = self.get_dinov2().cuda()

        self.feat_fusion = self.get_fusion().cuda()

    def train(self):
        self.feat_fusion.train()
        self.encoder.train()
        self.head.train()

    def get_encoder(self):
        return encoder(hidden_dim=self.hidden_dim, out_dim=self.dim, multires=self.encoder_multires)

    def get_head(self):
        return head(in_dim=self.dim, hidden_dim=self.hidden_dim)

    def get_dinov2(self):
        model = DINO2SEG(img_h=self.img_h, img_w=self.img_w, num_cls=self.n_classes, edge=self.crop_edge, dim=self.dim)
        sd = torch.load(self.pretrained_model_path, map_location=self.device)
        # The pretrained dinov2_replica.pth was trained with n_classes=52, so its
        # segmentation_conv.3.{weight,bias} have shape [52,16,3,3]/[52]. When CRCD
        # uses n_classes=4 these shapes diverge. load_state_dict(strict=False) does
        # NOT silence shape mismatches (only missing/unexpected keys), so we drop
        # the mismatched keys BEFORE the load. The final Conv2d is dead code under
        # use_gt_semantic=True (DINO2SEG.forward iterates segmentation_conv[0..2]
        # when mode='mapping'; see Mapper.py:476-481), so its random init is safe.
        model_sd = model.state_dict()
        dropped = []
        for k in list(sd.keys()):
            if k in model_sd and sd[k].shape != model_sd[k].shape:
                dropped.append((k, tuple(sd[k].shape), tuple(model_sd[k].shape)))
                del sd[k]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if dropped:
            print(f"[model_manager] DINO2SEG: dropped {len(dropped)} shape-mismatched keys "
                  f"from pretrained ckpt before load: {dropped}")
        if missing or unexpected:
            print(f"[model_manager] DINO2SEG load_state_dict — "
                  f"missing={list(missing)} unexpected={list(unexpected)}")
        return model

    def set_mode_feature(self):
        self.cnn.mode = 'mapping'

    def set_mode_result(self):
        self.cnn.mode = 'train'

    def get_fusion(self):
        return feature_fusion(dim=self.dim, hidden_dim=self.hidden_dim)

    def get_share_memory(self):
        self.feat_fusion.share_memory()
        self.encoder.share_memory()
        self.head.share_memory()
        self.cnn.share_memory()

        return self