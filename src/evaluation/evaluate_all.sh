python src/evaluation/evaluate_all.py \
   --test_data_path /home/weronika/Documents/masters/sem3/data_science_lab/DrTrack/evaluation/03_features/paired_patches_9_frames_1848_droplets_40x40.npy \
   --sample_size 1848 \
   --validation_batch_size 128 \
   --embed_dim 20 \
   --checkpoint_path ./checkpoints/2023-12-04_21-08-07_empty_droplets_dim_20.pth \
   --use_dapi