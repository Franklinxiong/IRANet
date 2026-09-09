from metrics import *
import os
import numpy as np

from sys import argv

script, exp_name = argv


pred_dir = ""
gt_dir = ""
db_name = 'glass_depth'

for sub_dir in sorted(os.listdir(pred_dir)):
    if 0 == len(os.listdir(os.path.join(pred_dir, sub_dir))):
        continue
    iou_l = []
    acc_l = []
    mae_l = []
    ber_l = []
    f_measure_p_l = []
    f_measure_r_l = []
    precision_record, recall_record = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
    for name in os.listdir(os.path.join(pred_dir, sub_dir, db_name)):
        gt = get_gt_mask(name, gt_dir)
        normalized_pred = get_normalized_predict_mask(name, os.path.join(pred_dir, sub_dir, db_name))
        binary_pred = get_binary_predict_mask(name, os.path.join(pred_dir, sub_dir, db_name))

        if normalized_pred.ndim == 3:
            normalized_pred = normalized_pred[:, :, 0]
        if binary_pred.ndim == 3:
            binary_pred = binary_pred[:, :, 0]

        acc_l.append(accuracy_mirror(binary_pred, gt))
        iou_l.append(compute_iou(binary_pred, gt))
        mae_l.append(compute_mae(normalized_pred, gt))
        ber_l.append(compute_ber(binary_pred, gt))

        pred = (255 * normalized_pred).astype(np.uint8)
        gt = (255 * gt).astype(np.uint8)
        p, r = cal_precision_recall(pred, gt)
        for idx, data in enumerate(zip(p, r)):
            p, r = data
            precision_record[idx].update(p)
            recall_record[idx].update(r)
    log = ('%s:  mae: %3f, ber: %3f, acc: %3f, iou: %3f, f_measure: %3f' % (os.path.join(pred_dir, sub_dir), np.mean(mae_l), np.mean(ber_l), np.mean(acc_l), np.mean(iou_l), cal_fmeasure([precord.avg for precord in precision_record], [rrecord.avg for rrecord in recall_record])))
           
    print(log)
    # print(tabulate(final_results, headers=table_header, tablefmt='grid'))
    with open('results_oldeval.txt', 'a') as w:
        w.write(log)
        w.write('\n')
