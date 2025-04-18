#!/usr/bin/env bash
    # Launch two local TensorFlow workers for smoke‑testing.
    HOST=127.0.0.1; PORT0=12345; PORT1=12346
    for IDX in 0 1; do
      export TF_CONFIG=$(cat <<JSON
    {"cluster":{"worker":["$HOST:$PORT0","$HOST:$PORT1"]},
     "task":{"type":"worker","index":$IDX}}
JSON
)
      echo "▶ worker $IDX"; CUDA_VISIBLE_DEVICES=$IDX python -u src/train_vae.py &
      PORT0=$PORT1
    done
    wait
