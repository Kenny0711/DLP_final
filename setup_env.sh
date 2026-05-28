#!/usr/bin/env bash
# Run once before training: source setup_env.sh
# 訓練前先執行一次：source setup_env.sh

# Add SUMO binary to PATH
SUMO_BIN=$(find "$HOME" -path "*/sumo/bin/sumo" -type f 2>/dev/null | head -1)
if [ -n "$SUMO_BIN" ]; then
    export PATH="$(dirname "$SUMO_BIN"):$PATH"
    echo "[setup] SUMO found: $SUMO_BIN"
else
    echo "[setup] WARNING: sumo binary not found"
fi

# Verify
which sumo && sumo --version 2>&1 | head -1
python3 -c "import torch; print('[setup] torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python3 -c "import traci; print('[setup] traci OK')"
