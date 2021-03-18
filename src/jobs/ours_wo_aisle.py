python train.py -cpk ours_wo_aisle -tb 1 -exp 1 -speaker '["oliver"]' -save_dir save/ours_wo_aisle/oliver -num_cluster 8 -model JointLateClusterSoftTransformer12_G -fs_new '15' -modalities '["pose/normalize", "text/tokens", "audio/log_mel_400"]' -gan 1 -loss L1Loss -window_hop 5 -render 0 -batch_size 32 -num_epochs 100 -stop_thresh 3 -overfit 0 -early_stopping 1 -dev_key dev_spatialNorm -feats '["pose", "velocity", "speed"]' -note ours_wo_aisle -dg_iter_ratio 1 -repeat_text 0 -num_iters 100 -min_epochs 50 -num_training_iters 400 -optim AdamW -lr 0.0001 -optim_separate 3e-05
