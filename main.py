import argparse

from feature_extractor_I3D.video2frame_split import run_I3D_frame_split
from feature_extractor_I3D.i3d_extract import run_I3D_feature_extraction
from reference.make_list import run_generate_list
from reference.make_gt import run_generate_gt
from anomaly_detector_PEL.vad_pel import run_vad_pel
from result.make_result import run_generate_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default="UCF_Testing", type=str,
                        help="Name of your video data in dataset folder")
    parser.add_argument('--level', default=2, type=int, choices=[1, 2],
                        help="Structure of your video data folder. If your dataset include subfolder, set this parameter to 2")
    parser.add_argument('--label', default=None, type=str,
                        help="If you have label of anomalies, please write the path of your label file")

    args = parser.parse_args()

    DIR_video = f"./dataset/video/{args.input}"
    DIR_frame = f"./dataset/frame/{args.input}"
    DIR_feature = f"./dataset/feature/{args.input}"
    DIR_list = f"./reference/{args.input}.list"
    DIR_result = f"./result/{args.input}.xlsx"

    '''  1. 입력 비디오를 프레임 단위의 이미지로 분할  '''
    run_I3D_frame_split(DIR_video,DIR_frame,args.level)

    '''  2. 프레임을 활용하여 비디오 피처 추출  '''
    run_I3D_feature_extraction(DIR_frame,DIR_feature,args.level)

    '''  3. 비디오 피처 리스트 파일 생성  '''
    run_generate_list(args.input, DIR_feature)

    '''  3-1. 라벨이 존재할 경우 Grount Truth 파일 생성  '''
    DIR_gt = None
    if args.label is not None:
        DIR_label = f"./reference/{args.label}"
        run_generate_gt(args.input, DIR_feature, DIR_label)
        DIR_gt = f"./reference/{args.input}_gt.npy"

    '''  4. Anomaly Detection 시행 '''
    video_list, pred_list = run_vad_pel(DIR_feature, DIR_list, DIR_gt)

    '''  5. 결과 엑셀파일 생성  '''
    run_generate_result(DIR_result, video_list, pred_list)
