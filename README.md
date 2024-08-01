# End2End_VAD
비디오 데이터에 대하여 End-to-End 로 이상상황 감지를 수행하기 위한 프로젝트입니다.
아래의 가이드에 따라 영상 이상상황 감지 및 이상 스코어를 확인할 수 있습니다.

## Environment
- 3.8 버전 이상의 python 가상환경이 필요합니다.
```
conda create -n <가상환경이름> python=3.8
```
- PEL 모델 활용을 위해 아래 명령와 같은 pytorch 1.8 버전 설치가 필요합니다.
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```
- 그 외 라이브러리는 아래의 명령어로 설치할 수 있습니다.
```
pip install -r requirements.txt
```

## Dataset
- 자체적인 비디오 데이터셋을 ./dataset/video/ 경로에 위치시키면 됩니다.
- 아래의 구조 예시와 같이, 데이터셋 폴더에 하위 폴더가 존재할 경우, main.py 함수 실행 시 --level=2 로 설정하면 됩니다.
```
├── dataset/
│    └── video/
│          └── UCF_Testing/
│                      └── Abuse/
│                            ├── Abuse028_x264.mp4
│                            ├── Abuse030_x264.mp4
│                      └── Arrest/
│                            ├── Arrest001_x264.mp4
│                            ├── ...

```

- 아래의 구조 예시와 같이, 데이터셋 폴더에 비디오 파일이 바로 위치한 경우, main.py 함수 실행 시 --level=1 로 설정하면 됩니다.
```
├── dataset/
│    └── video/
│          └── UCF_Testing/
│                      ├── Abuse028_x264.mp4
│                      ├── Abuse030_x264.mp4
│                      ├── Arrest001_x264.mp4
│                      ├── ...
```
- 비디오 데이터에 대한 이상상황 라벨이 존재할 경우, 성능 평가를 위한 Gount Truth 생성을 위해 아래와 같이 라벨 파일을 위치시켜야 합니다.
- 라벨 파일의 형식은 'Temporal_Anomaly_Annotation_for_Testing_Videos.txt' 와 같이 프레임 단위여야합니다.
```
├── reference/
│          ├── Temporal_Anomaly_Annotation_for_Testing_Videos.txt
```

## Inference
- main.py 파일을 실행하여 End-to-End 로 Video Anomaly Detection 작업을 수행할 수 있습니다.
(아래의 데이터셋 폴더명대로 프레임, 피처, gt, 결과 엑셀 파일 등이 생성됩니다. 원하는 결과명에 따라 폴더명을 변경 후 적용해주시기 바랍니다.)
```
python main.py --input <데이터셋 폴더 이름> --level <하위폴더가 있을경우 2, 아닐경우 1> --label <라벨 파일 이름(필요한 경우에만 입력)>
```
- 아래의 예시 명령어를 활용하여 UCF-Crime Testing 데이터셋에 대한 이상상황 감지를 진행할 수 있습니다.
```
python main.py --input UCF_Testing --level 2 --label Temporal_Anomaly_Annotation_for_Testing_Videos.txt
```
- 이상상황 감지 결과 엑셀파일은 ./results 폴더에 생성됩니다.

## Reference
- Dataset : [UCF-Crime](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/)
- Feature Extraction : [I3D implementation from UR-DMU](https://github.com/henrryzh1/UR-DMU/tree/master/feature_extract)
- Video Anomaly Detection : [PEL](https://github.com/yujiangpu20/pel4vad)
