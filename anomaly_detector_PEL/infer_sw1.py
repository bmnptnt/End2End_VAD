from sklearn.metrics import auc, roc_curve, confusion_matrix, precision_recall_curve, f1_score
import numpy as np
import torch
import time
from tqdm import tqdm
from utils import fixed_smooth, slide_smooth

import matplotlib.pyplot as plt
import os
STRIDE=1
def cal_false_alarm(gt, preds, threshold=0.5):
    preds = list(preds.cpu().detach().numpy())
    gt = list(gt.cpu().detach().numpy())

    # preds = np.repeat(preds, 16)
    preds[preds < threshold] = 0
    preds[preds >= threshold] = 1
    tn, fp, fn, tp = confusion_matrix(gt, preds, labels=[0, 1]).ravel()

    far = fp / (fp + tn)

    return far
def infer_func(model, dataloader, gt, logger, cfg):
    st = time.time()
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).cuda()
        normal_preds = torch.zeros(0).cuda()
        normal_labels = torch.zeros(0).cuda()
        gt_tmp = torch.tensor(gt.copy()).cuda()
        frames = 0

        video_list=[]
        frame_list=[]

        for i, (v_input, name, feat_path) in enumerate(dataloader):
            video_name = os.path.basename(feat_path[0]).split('_x264')[0]
            video_list.append(video_name)

            frames += v_input.shape[1]
            v_input = v_input.float().cuda(non_blocking=True)
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)

            if v_input.shape[1] > 5000:
                logits_list = []
                num_seg = v_input.shape[1] // 5000 + 1
                v_input_segs = torch.chunk(v_input, num_seg, dim=1)
                for i in v_input_segs:
                    seq_len = torch.sum(torch.max(torch.abs(i), dim=2)[0] > 0, 1)
                    logits_elem,_=model(i,seq_len)
                    logits_list.append(logits_elem)

                logits = torch.cat(logits_list, dim=1)
            else:
                logits,_ = model(v_input,seq_len)

            # logits, _ = model(v_input, seq_len)
            logits = torch.mean(logits, 0)
            logits = logits.squeeze(dim=-1)

            seq = len(logits)
            frame_list.append(seq)
            if cfg.smooth == 'fixed':
                logits = fixed_smooth(logits, cfg.kappa)
            elif cfg.smooth == 'slide':
                logits = slide_smooth(logits, cfg.kappa)
            else:
                pass
            logits = logits[:seq]

            pred = torch.cat((pred, logits))
            # labels = gt_tmp[: seq_len[0]*16]
            labels = gt_tmp[: seq_len[0]]
            if torch.sum(labels) == 0:
                normal_labels = torch.cat((normal_labels, labels))
                normal_preds = torch.cat((normal_preds, logits))
            # gt_tmp = gt_tmp[seq_len[0]*16:]
            gt_tmp = gt_tmp[seq_len[0]:]

        time_elapsed = time.time() - st
        pred_np=pred.cpu().detach().numpy()
        pred = list(pred_np)
        # far = cal_false_alarm(normal_labels, normal_preds)
        # fpr, tpr, _ = roc_curve(list(gt), np.repeat(pred, 16))
        fpr, tpr, _ = roc_curve(list(gt), pred)
        roc_auc = auc(fpr, tpr)
        # pre, rec, th = precision_recall_curve(list(gt), np.repeat(pred, 16))
        pre, rec, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(rec, pre)

        f1_scores=[]
        for i in tqdm(range(0,len(th),1000)):
            frame_classify=[1 if p >=th[i] else 0 for p in pred]
            f1= f1_score(list(gt),frame_classify)
            f1_scores.append(f1)
        max_f1_index=f1_scores.index(max(f1_scores))
        best_thd=th[max_f1_index*1000]
        best_f1=f1_scores[max_f1_index]
        print(f"Best F1-Score : {max(f1_scores)} , Best Threshold : {best_thd} , Best THD Index : {max_f1_index * 1000}")

        # frame_classify = [1 if pred >= 0.49 else 0 for pred in np.repeat(pred, 16)] #16seg
        frame_classify = [1 if pred >= 0.045 else 0 for pred in pred] #sw1
        # frame_classify = [1 if pred >= 0.007 else 0 for pred in pred]  # sw8

        f1 = f1_score(list(gt), frame_classify)

    start_frame = -1
    end_frame = 0

    print(f"Generate anomaly score graphs..")
    for i in tqdm(range(len(frame_list))):
        start_frame = end_frame
        end_frame = end_frame + frame_list[i]
        video = video_list[i]
        score_fig = plt.figure()

        frame_num = np.arange(1, frame_list[i] + 1)
        plt.plot(frame_num, pred_np[start_frame:end_frame], label=video, color='b')
        plt.fill_between(frame_num, 0, gt[start_frame:end_frame], color='r', alpha=0.3, label='Ground Truth (anomaly)')

        plt.xlabel("Frame")
        plt.ylabel("Score")
        plt.title(f"PEL : {video} Anomaly Scores")
        plt.savefig(f'./graph/sw1/{video}.pdf')
        plt.close(score_fig)


    print(gt.shape,pred_np.shape)
    print(f"F1-Score : {f1}")

    logger.info('offline AUC:{:.4f} AP:{:.4f}| Complete in {:.0f}m {:.0f}s\n'.format(
        roc_auc, pr_auc, time_elapsed // 60, time_elapsed % 60))
