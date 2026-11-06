#!/usr/bin/env bash
# usage: run-llamampere.sh <context> [--exact]
CTX=${1:-32768}; shift
export GGML_MOE_CPU_SPLIT=0
export LLAMA_SERVER_BIN=$HOME/llamAmpere/build-sm86/bin/llama-server
export FLASH_NEXT_DRAFTER=/models/drafter/mtp-Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2S-B.gguf
exec $HOME/llamAmpere/scripts/flash-next-server.sh "$@" \
  -m /models/swift-fn/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  -c $CTX --kv-gpu-tokens 32768 \
  --jinja --parallel 1 --cache-ram 0 -t ${T:-30} -td ${T:-30} --host 127.0.0.1 --port 8080
