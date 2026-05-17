mkdir -p ./output/AVI_track2 ./log

nohup python -u main.py \
    --output_model ./output/AVI_track2/best_model.pth \
    --test_model ./output/AVI_track2/best_model.pth \
    --test_output_csv ./output/AVI_track2/submission.csv \
    --train_csv ./data-2026/train_data.csv \
    --val_csv ./data-2026/val_data.csv \
    --test_csv ./data-2026/test_data.csv \
    --question q1 q2 q3 q4 q5 q6 \
    --label_col g_level \
    --metadata_cols gender age education work_experience H_self E_self A_self C_self \
    --video_dim 1536 \
    --video_dir ./features \
    --audio_dim 1536 \
    --audio_dir ./features \
    --text_dim 1536 \
    --text_dir ./features \
    --target_dim 3 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_epochs 50 \
    --log_dir ./log \
    > ./output/AVI_track2/track2_train.log 2>&1 &
