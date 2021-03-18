python train.py -cpk gesticulator -tb 1 -exp 1 -speaker '["oliver"]' -save_dir save/gesticulator/oliver -num_cluster 8 -model GestLate_G -fs_new '15' -modalities '["pose/normalize", "text/bert", "audio/log_mel_512", "audio/silence"]' -input_modalities '["text/bert", "audio/log_mel_512"]' -output_modalities '["pose/normalize"]' -filler 1 -gan 0 -loss MSELoss -window_hop 5 -render 0 -batch_size 32 -num_epochs 100 -stop_thresh 3 -overfit 0 -early_stopping 1 -dev_key dev_spatialNorm -feats '["pose", "velocity", "speed"]' -note gesticulator -dg_iter_ratio 1 -repeat_text 1 -num_iters 100 -num_training_iters 400 -min_epochs 50 -optim AdamW -lr 0.0001