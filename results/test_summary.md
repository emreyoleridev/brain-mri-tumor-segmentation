|                        |   U-Net (thr 0.5, no post-proc.) |   U-Net + post-proc. (thr 0.7, min area 100) |
|:-----------------------|---------------------------------:|---------------------------------------------:|
| global_dice            |                           0.814  |                                       0.8219 |
| global_iou             |                           0.6864 |                                       0.6976 |
| mean_dice_tumor_slices |                           0.7781 |                                       0.7737 |
| mean_iou_tumor_slices  |                           0.6814 |                                       0.6775 |
| mean_dice_all_slices   |                           0.7794 |                                       0.8395 |
| mean_patient_dice      |                           0.7681 |                                       0.7757 |
| pixel_precision        |                           0.7905 |                                       0.8329 |
| pixel_recall           |                           0.8391 |                                       0.8111 |
| detect_accuracy        |                           0.8414 |                                       0.9001 |
| detect_sensitivity     |                           0.9542 |                                       0.9458 |
| detect_specificity     |                           0.78   |                                       0.8753 |
| detect_precision       |                           0.7025 |                                       0.805  |