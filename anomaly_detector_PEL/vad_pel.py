from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import time
import numpy as np
import random
from .configs import build_config
from .utils import setup_seed
from .log import get_logger

from .model import XModel
from .dataset import *

from .infer import infer_func
import argparse
import copy

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_checkpoint(model, ckpt_path, logger):
    if os.path.isfile(ckpt_path):
        logger.info('loading pretrained checkpoint from {}.'.format(ckpt_path))
        weight_dict = torch.load(ckpt_path)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    logger.info('{} size mismatch: load {} given {}'.format(
                        name, param.size(), model_dict[name].size()))
            else:
                logger.info('{} not found in model dict.'.format(name))
    else:
        logger.info('Not found pretrained checkpoint file.')


def main(cfg):
    logger = get_logger(cfg.logs_dir)
    setup_seed(cfg.seed)
    # logger.info('Config:{}'.format(cfg.__dict__))

    test_data = VAD_Dataset(cfg)
    test_loader = DataLoader(test_data, batch_size=cfg.test_bs, shuffle=False,
                             num_workers=cfg.workers, pin_memory=True)
    model = XModel(cfg)
    if cfg.gt is not None:
        gt = np.load(cfg.gt)
    else:
        gt = None

    device = torch.device("cuda")
    model = model.to(device)

    param = sum(p.numel() for p in model.parameters())
    logger.info('total params:{:.4f}M'.format(param / (1000 ** 2)))

    logger.info('Test Mode')
    if cfg.ckpt_path is not None:
        load_checkpoint(model, cfg.ckpt_path, logger)
    else:
        logger.info('infer from random initialization')
    video_list, pred_list = infer_func(model, test_loader, gt, logger, cfg)

    return video_list, pred_list


def run_vad_pel(feat_dir, list_dir, gt_dir, mode='UCF', ckpt='ucf__8636.pkl'):
    cfg = build_config(feat_dir, gt_dir, list_dir, mode, ckpt)
    video_list, pred_list = main(cfg)
    return video_list, pred_list

