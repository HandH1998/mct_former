# train MCTformerV2 on aug train set
python main.py --model deit_small_MCTformerV2_patch16_224 \
    --batch-size 64 \
    --data-set VOC12 \
    --img-list datasets/voc12 \
    --label-file-path datasets/voc12/cls_labels.npy \
    --data-path datasets/voc12/VOCdevkit/VOC2012 \
    --output_dir models/voc12/MCTformer_v2/cls \
    --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth # change to our pretrained deit 

# generate class-specific localization maps on aug train set
python main.py --model deit_small_MCTformerV2_patch16_224 \
    --data-set VOC12MS \
    --scales 1.0 \
    --img-list datasets/voc12 \
    --label-file-path datasets/voc12/cls_labels.npy \
    --data-path datasets/voc12/VOCdevkit/VOC2012 \
    --resume models/voc12/MCTformer_v2/cls/checkpoint_best.pth \
    --gen_attention_maps \
    --attention-type fused \
    --layer-index 3 \
    --visualize-cls-attn \
    --patch-attn-refine \
    --attention-dir results/voc12/MCTformer_v2/attn-patchrefine \
    --cam-npy-dir results/voc12/MCTformer_v2/attn-patchrefine-npy \
    --out-crf results/voc12/MCTformer_v2/attn-patchrefine-npy-crf 

# evaluate the generated class-specific localization maps on train set
python evaluation.py --list datasets/voc12/train_id.txt \
    --gt_dir datasets/voc12/VOCdevkit/VOC2012/SegmentationClassAug \
    --logfile results/voc12/MCTformer_v2/attn-patchrefine-npy/eval_log_train.txt \
    --type npy \
    --curve True \
    --predict_dir results/voc12/MCTformer_v2/attn-patchrefine-npy \
    --comment "voc12_train_1464"

# evaluate the generated class-specific localization maps on aug train set
python evaluation.py --list datasets/voc12/train_aug_id.txt \
    --gt_dir datasets/voc12/VOCdevkit/VOC2012/SegmentationClassAug \
    --logfile results/voc12/MCTformer_v2/attn-patchrefine-npy/eval_log_train_aug.txt \
    --type npy \
    --curve True \
    --predict_dir results/voc12/MCTformer_v2/attn-patchrefine-npy \
    --comment "voc12_aug_train_10582"

# train AffinityNet on aug train set
python psa/train_aff.py --weights models/res38/res38_cls.pth \
    --voc12_root datasets/voc12/VOCdevkit/VOC2012/ \
    --train_list datasets/voc12/train_aug_id.txt \
    --save_path models/voc12/MCTformer_v2/aff \
    --la_crf_dir results/voc12/MCTformer_v2/attn-patchrefine-npy-crf_1 \
    --ha_crf_dir results/voc12/MCTformer_v2/attn-patchrefine-npy-crf_12 


# get final pseudo pixel labels on aug train set
python psa/infer_aff.py --weights models/voc12/MCTformer_v2/aff/resnet38_aff.pth \
    --infer_list datasets/voc12/train_aug_id.txt \
    --cam_dir results/voc12/MCTformer_v2/attn-patchrefine-npy \
    --voc12_root datasets/voc12/VOCdevkit/VOC2012/ \
    --out_rw results/voc12/MCTformer_v2/pgt-psa-rw 

# evaluate the final pseudo pixel labels on train set
python evaluation.py --list datasets/voc12/train_id.txt \
    --gt_dir datasets/voc12/VOCdevkit/VOC2012/SegmentationClassAug \
    --logfile results/voc12/MCTformer_v2/pgt-psa-rw/eval_log_train.txt \
    --type png \
    --predict_dir results/voc12/MCTformer_v2/pgt-psa-rw \
    --comment "voc12_train_1464"

# evaluate the final pseudo pixel labels on aug train set
python evaluation.py --list datasets/voc12/train_aug_id.txt \
    --gt_dir datasets/voc12/VOCdevkit/VOC2012/SegmentationClassAug \
    --logfile results/voc12/MCTformer_v2/pgt-psa-rw/eval_log_train_aug.txt \
    --type png \
    --predict_dir results/voc12/MCTformer_v2/pgt-psa-rw \
    --comment "voc12_aug_train_10582"

# train seg network on aug train set
python seg/train_seg.py --network resnet38_seg \
    --num_epochs 30 \
    --seg_pgt_path results/voc12/MCTformer_v2/pgt-psa-rw \
    --init_weights models/res38/res38_cls.pth\
    --save_path  models/voc12/MCTformer_v2/seg \
    --list_path datasets/voc12/train_aug_id.txt \
    --img_path datasets/voc12/VOCdevkit/VOC2012/JPEGImages \
    --num_classes 21 \
    --batch_size 4

# evaluate seg network without post-processing
python seg/infer_seg.py --weights models/voc12/MCTformer_v2/seg/resnet38_seg.pth \
    --network resnet38_seg \
    --list_path datasets/voc12/val_id.txt \
    --gt_path datasets/voc12/VOCdevkit/VOC2012/SegmentationClassAug \
    --img_path datasets/voc12/VOCdevkit/VOC2012/JPEGImages\
    --save_path results/voc12/MCTformer_v2/val_ms_crf_no_post \
    --save_path_c results/voc12/MCTformer_v2/val_ms_crf_c_no_post \
    --scales 1.0

# evaluate seg network with crf post-processing
python seg/infer_seg.py --weights models/voc12/MCTformer_v2/seg/resnet38_seg.pth \
    --network resnet38_seg \
    --list_path datasets/voc12/val_id.txt \
    --gt_path datasets/voc12/VOCdevkit/VOC2012/SegmentationClassAug \
    --img_path datasets/voc12/VOCdevkit/VOC2012/JPEGImages\
    --save_path results/voc12/MCTformer_v2/val_ms_crf \
    --save_path_c results/voc12/MCTformer_v2/val_ms_crf_c \
    --scales 0.5 0.75 1.0 1.25 1.5 \
    --use_crf True \
    2>&1 | tee models/voc12/MCTformer_v2/seg/resnet38_seg_infer.log

