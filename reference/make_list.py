import os
from glob import glob




def run_generate_list(seq_name,feat_dir):
    with open(f"./reference/{seq_name}.list",'w') as list_file:
        feat_list=sorted(os.listdir(f"{feat_dir}"))
        for f_name in feat_list:
            list_file.write(f"{f_name}\n")

