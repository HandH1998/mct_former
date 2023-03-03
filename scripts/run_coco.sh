# train MCTformerV2 on train set
python3 main.py --model deit_small_MCTformerV2_patch16_224 \
    --batch-size 64 \
    --data-set COCO \
    --img-list datasets/coco \
    --label-file-path  datasets/coco/cls_labels.npy \
    --data-path datasets/coco \
    --output_dir models/coco/MCTformer_v2/cls \
    --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth

# generate class-specific localization maps on train set
python3 main.py --model deit_small_MCTformerV2_patch16_224 \
    --data-set COCOMS \
    --scales 1.0 \
    --img-list datasets/coco \
    --label-file-path datasets/coco/cls_labels.npy \
    --data-path datasets/coco \
    --resume models/coco/MCTformer_v2/cls/checkpoint_best.pth \
    --gen_attention_maps \
    --attention-type fused \
    --layer-index 3 \
    --visualize-cls-attn \
    --patch-attn-refine \
    --attention-dir results/coco/MCTformer_v2/attn-patchrefine \
    --cam-npy-dir results/coco/MCTformer_v2/attn-patchrefine-npy \
    --out-crf results/coco/MCTformer_v2/attn-patchrefine-npy-crf

# evaluate the generated class-specific localization maps on train set
python3 evaluation.py --list datasets/coco/train_id.txt \
    --gt_dir datasets/coco/SegmentationClass \
    --logfile results/coco/MCTformer_v2/attn-patchrefine-npy/eval_log_train.txt \
    --type npy \
    --curve True \
    --predict_dir results/coco/MCTformer_v2/attn-patchrefine-npy \
    --comment "coco_train_82081"

# train AffinityNet on train set
python3 psa/train_aff.py --weights models/res38/res38_cls.pth \
    --voc12_root datasets/coco \
    --train_list datasets/coco/train_id.txt \
    --save_path models/coco/MCTformer_v2/aff \
    --la_crf_dir results/coco/MCTformer_v2/attn-patchrefine-npy-crf_1 \
    --ha_crf_dir results/coco/MCTformer_v2/attn-patchrefine-npy-crf_12 

# get final pseudo pixel labels on train set
python3 psa/infer_aff.py --weights models/coco/MCTformer_v2/aff/resnet38_aff.pth \
    --infer_list datasets/coco/train_id.txt \
    --cam_dir results/coco/MCTformer_v2/attn-patchrefine-npy \
    --voc12_root datasets/coco \
    --out_rw results/coco/MCTformer_v2/pgt-psa-rw 

# evaluate the final pseudo pixel labels on train set
python3 evaluation.py --list datasets/coco/train_id.txt \
    --gt_dir datasets/coco/SegmentationClass \
    --logfile results/coco/MCTformer_v2/pgt-psa-rw/eval_log_train.txt \
    --type png \
    --predict_dir results/coco/MCTformer_v2/pgt-psa-rw \
    --comment "coco_train_82081"

# train seg network on aug train set
python3 seg/train_seg.py --network resnet38_seg \
    --num_epochs 30 \
    --seg_pgt_path results/coco/MCTformer_v2/pgt-psa-rw \
    --init_weights models/res38/res38_cls.pth\
    --save_path  models/coco/MCTformer_v2/seg \
    --list_path datasets/coco/train_id.txt \
    --img_path datasets/coco/JPEGImages \
    --num_classes 81 \
    --batch_size 4

# evaluate seg network without post-processing
python3 seg/infer_seg.py --weights models/coco/MCTformer_v2/seg/resnet38_seg.pth \
    --network resnet38_seg \
    --list_path datasets/coco/val_id.txt \
    --gt_path datasets/coco/SegmentationClass \
    --img_path datasets/coco/JPEGImages \
    --save_path results/coco/MCTformer_v2/val_ms_crf_no_post \
    --save_path_c results/coco/MCTformer_v2/val_ms_crf_c_no_post \
    --scales 1.0

# evaluate seg network with crf post-processing
python3 seg/infer_seg.py --weights models/coco/MCTformer_v2/seg/resnet38_seg.pth \
    --network resnet38_seg \
    --list_path datasets/coco/val_id.txt \
    --gt_path datasets/coco/SegmentationClass \
    --img_path datasets/coco/JPEGImages \
    --save_path results/coco/MCTformer_v2/val_ms_crf \
    --save_path_c results/coco/MCTformer_v2/val_ms_crf_c \
    --scales 0.5 0.75 1.0 1.25 1.5 \
    --use_crf True \
    2>&1 | tee models/coco/MCTformer_v2/seg/resnet38_seg_infer.log