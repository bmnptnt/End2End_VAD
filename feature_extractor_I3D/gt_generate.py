import os
from glob import glob
import numpy as np
# frames=0
# for i,f in enumerate(glob(f"./UCF_ten/*")):
#     feat=np.load(f)
#     frames+=feat.shape[1]
# print(frames)
STRIDE=8
gt_txt=open('Temporal_Anomaly_Annotation_for_Testing_Videos.txt')
gt=[]
for i in gt_txt:
    video_gt=i.split('\n')[0]
    video_name=video_gt.split('  ')[0]
    feature=np.load(f"./UCF_ten_sw8/{video_name.split('.')[0]}_i3d.npy")
    num_frame=feature.shape[1]*STRIDE
    frame_gt=np.zeros(num_frame)
    if video_gt.split('  ')[2] !='-1' and video_gt.split('  ')[3] !='-1' :
        frame_gt[int(video_gt.split('  ')[2]):int(video_gt.split('  ')[3])+1]=1
    if video_gt.split('  ')[4] !='-1' and video_gt.split('  ')[5] !='-1' :
        frame_gt[int(video_gt.split('  ')[4]):int(video_gt.split('  ')[5])+1]=1
    gt.append(frame_gt)
all_gt=np.concatenate(gt)
print(f"Num of gt frames : {all_gt.shape}")
np.save("ucf_gt_sw8.npy",all_gt)