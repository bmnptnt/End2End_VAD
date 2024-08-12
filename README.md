# End2End_VAD
This repository is End-to-End Video Anomaly Detection project.
Follow the instructions below to detect anomalies from video and check abnormal scores.

*※ 한국어 가이드는 README_kr.md 파일을 참조하시기 바랍니다.*
## Environment
- You must install a python virtual environment of version >= 3.8 
```
conda create -n <env name> python=3.8
```
- In order to utilize the PEL as anomaly detector, installation of pytorch 1.8 version is required by follwing command.
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```
- Other libraries can be installed with the following command.
```
pip install -r requirements.txt
```

## Dataset
- Place your custom video dataset in the './dataset/video/' path.
- For example, When you apply a dataset called 'UCF_Testing', if dataset folder include subfolder, you shuld set '--level 2' when running 'main.py'.
```
├── dataset/
│    └── video/
│          └── UCF_Testing/
│                      ├── Abuse/
│                      │     ├── Abuse028_x264.mp4
│                      │     ├── Abuse030_x264.mp4
│                      ├── Arrest/
│                      │     ├── Arrest001_x264.mp4
│                      │     ├── ...

```

- If dataset forlder include video files directly, you should set '--level 1' when running 'main.py'.
```
├── dataset/
│    └── video/
│          └── UCF_Testing/
│                      ├── Abuse028_x264.mp4
│                      ├── Abuse030_x264.mp4
│                      ├── Arrest001_x264.mp4
│                      ├── ...
```
- (optional)If you have anomaly labels of video dataset, Place you label file in './reference' for performance evaluation.
- The format of label file must be consist of label for frame such as 'Temporal_Anomaly_Annotation_for_Testing_Videos.txt'.
```
├── reference/
│          ├── Temporal_Anomaly_Annotation_for_Testing_Videos.txt
```

## Inference
- By running 'main.py', You canb implement End-to-End Video Anomaly Detection.
(frame, feature, gt, result Excel file, etc. are generated as 'Name of dataset folder' below. Please change the folder name according to the desired result name and apply it.)
```
python main.py --input <Name of dataset folder> --level <If datset folder include sufolder 2, If datset folder include videos directly 1> --label <(optional)Name of label file>
```
- By following example commands, You can implement Video Anomaly Detection in the UCF-Crime Testing dataset.
```
python main.py --input UCF_Testing --level 2 --label Temporal_Anomaly_Annotation_for_Testing_Videos.txt
```
- Excel file of anomaly detection is generated in './results' folder.

## Reference
- Dataset : [UCF-Crime](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/)
- Feature Extraction : [I3D implementation from UR-DMU](https://github.com/henrryzh1/UR-DMU/tree/master/feature_extract)
- Video Anomaly Detection : [PEL](https://github.com/yujiangpu20/pel4vad)
