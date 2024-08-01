import pandas as pd
import numpy as np

def convert_seconds(seconds):
    hours = seconds // 3600  # 1시간은 3600초
    minutes = (seconds % 3600) // 60  # 1시간을 제외한 나머지 초를 60으로 나눔
    remaining_seconds = seconds % 60  # 남은 초
    return hours, minutes, remaining_seconds
def run_generate_result(result_dir,video_list,pred_list):
    results=[]
    for i in range(len(video_list)):
        for j in range(0,len(pred_list[i]),30):
            h, m, s=convert_seconds(j//30)
            results.append([video_list[i],pred_list[i][j],f"{h:02d}:{m:02d}:{s:02d}"])

    results_df=pd.DataFrame(results, columns=['Video file name', 'Anomaly score', 'Timestamp (second)'])
    results_df.to_excel(result_dir, index=False,sheet_name='Sheet1')

    print(f"Successfully generated result file to {result_dir}")
