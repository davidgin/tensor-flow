#!/usr/bin/env bash
# Launch two local workers for the modular VAE example
HOST=127.0.0.1
PORT0=12345
PORT1=12346
WORKERS=("$HOST:$PORT0" "$HOST:$PORT1")

for IDX in 0 1; do
  export TF_CONFIG=$(cat <<EOF
{
  "cluster": { "worker": ["${WORKERS[0]}", "${WORKERS[1]}"] },
  "task":    { "type":"worker", "index": $IDX }
}
EOF
  )
  echo "▶ Worker $IDX on port ${WORKERS[$IDX]##*:}"
  CUDA_VISIBLE_DEVICES=$IDX python -u src/train.py &
done

wait
