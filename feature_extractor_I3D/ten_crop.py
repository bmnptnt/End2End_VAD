import os
from tqdm import tqdm
import numpy as np

def run_crop_feature(source,dst):
    features=os.listdir(source)
    for feature in tqdm(features):
        data=np.load(os.path.join(source,feature))
        for i in range(10):
            if i==0:
                np.save("{}/{}.npy".format(dst, feature.split("_i3d")[0]), data[i])
            else :
                np.save("{}/{}__{}.npy".format(dst, feature.split("_i3d")[0], i), data[i])

