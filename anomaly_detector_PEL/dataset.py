import torch.utils.data as data
from .utils import process_feat
import numpy as np
import os


class VAD_Dataset(data.Dataset):
    def __init__(self, cfg, transform=None):
        self.feat_prefix = cfg.feat_prefix
        self.list_file = cfg.test_list
        self.max_seqlen = cfg.max_seqlen
        self.tranform = transform
        self._parse_list()

    def _parse_list(self):
        self.list = list(open(self.list_file))

    def __getitem__(self, index):
        # video_name = self.list[index].strip('\n').split('/')[-1][:-4]
        feat_path = os.path.join(self.feat_prefix, self.list[index].strip('\n'))
        v_feat = np.array(np.load(feat_path), dtype=np.float32)
        if self.tranform is not None:
            v_feat = self.tranform(v_feat)
        return v_feat, feat_path

    def __len__(self):
        return len(self.list)


