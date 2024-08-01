import os
from glob import glob
import numpy as np


def run_generate_gt(seq_name,feat_dir,label_path,stride=16):
    gt_txt = open(label_path)
    gt = []
    for i in gt_txt:
        video_gt = i.split('\n')[0]
        video_name = video_gt.split('  ')[0]
        feature = np.load(f"./{feat_dir}/{video_name.split('.')[0]}_i3d.npy")
        num_frame = feature.shape[1] * stride
        frame_gt = np.zeros(num_frame)
        if video_gt.split('  ')[2] != '-1' and video_gt.split('  ')[3] != '-1':
            frame_gt[int(video_gt.split('  ')[2]):int(video_gt.split('  ')[3]) + 1] = 1
        if video_gt.split('  ')[4] != '-1' and video_gt.split('  ')[5] != '-1':
            frame_gt[int(video_gt.split('  ')[4]):int(video_gt.split('  ')[5]) + 1] = 1
        gt.append(frame_gt)
    all_gt = np.concatenate(gt)
    print(f"Num of gt frames : {all_gt.shape}")
    np.save(f"./reference/{seq_name}_gt.npy", all_gt)