python train.py --epochs 5 \
    --learning_rate 3e-5 \
    --batch_size 16 \
    --warmup_ratio 0.05 \
    --output_dir checkpoints \
    --logging_steps 500 \
    --save_steps 1500 \
    --num_workers 4