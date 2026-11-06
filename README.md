# RTX 3090 vs a 125B Model: Strata and llamAmpere, Tested

A six-year-old RTX 3090, 64 GB of system RAM and a 125-billion-parameter model.
Two engines, one machine, the same model files and the same prompts.

**Strata: 118 tok/s out of the box. llamAmpere: 51 tok/s at first, then 97 tok/s after a one-line fix.**

This repo has everything from the video: the setup, both llamAmpere fixes, the benchmark script and my raw results.

📺 Video: *[link]* · Channel: [Network Coder](https://www.youtube.com/@networkcoder)

---

## Scoreboard

| Round | Strata | llamAmpere (after fixes) |
|---|---|---|
| Short question, writing speed | **118.5 tok/s** | 96.7 tok/s |
| 100K-token prompt, time to read it | **44 s** (2,330 tok/s) | 2 min 46 s (617 tok/s) |
| 100K-token prompt, writing speed | **98.0 tok/s** | 81.0 tok/s |
| ~250K-token prompt | **~2 min to read, 89.9 tok/s writing** | Did not finish (out of VRAM) |
| Claim test: 100K in, 5,120 out, temp 1.0, seed 23 | **101.3 tok/s** writing, 2,322 tok/s reading | 96.3 tok/s writing, 614 tok/s reading |
| Setup | Tuned itself, no flags changed | Needed 2 changes |

llamAmpere's README claims **113 tok/s writing and ~2,751 tok/s reading** on a 100K prompt. Those numbers come from an **RTX 3090 Ti**; mine is a regular 3090 capped at 260 W.
The writing speed comes close to the claim; the reading speed doesn't.

Each round was run twice and the second run kept (the first run warms things up). Every prompt starts with a unique tag, so nothing is reused from cache.

---

## Test machine

| | |
|---|---|
| GPU | NVIDIA RTX 3090, 24 GB, power limit **260 W** (could not be raised) |
| CPU | AMD EPYC 7742, about 30 cores available |
| RAM | **64 GB** available to the engines |
| Model | Qwen3.8-Flash-Next, Swift 1.5 **IQ3_XXS** (~76 GB on disk, 125B parameters) |

> Hugging Face shows the model as "177B params". That count includes a large lookup table; the model itself is 125B.

Both engines used **the exact same model files**. Each used its own small "draft" helper, the way it's meant to run:
- Strata: its own draft layer (~5.2 GB, downloaded automatically)
- llamAmpere: `mtp-Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2S-B.gguf` (~1.5 GB)

---

## 1. Strata

Install (one line, Linux/macOS):

```bash
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

Preview what it will do first (changes nothing):

```bash
strata launch --plan --family qwen --model q2_0
```

Launch used in the test:

```bash
strata launch --model-path "/models/swift-fn" --model iq3_xxs --family swift \
  --context 32k --vision off --drafting mtp --tune quick --access local --port 8080
```

On first start it tunes itself: it measures your machine, picks settings, then measures again. Keep the machine quiet while it does this.
Loading 76 GB of model into 64 GB of RAM takes about a minute.

---

## 2. llamAmpere (v0.6.2)

Clone and build for the 3090 (sm_86), following the project's own build instructions:

```bash
git clone <llamAmpere repo URL> ~/llamAmpere
cd ~/llamAmpere
cmake -B build-sm86 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-sm86 -j
```

The project ships a launcher, `scripts/flash-next-server.sh`. It reads `data/moe/flash-next.env` (settings made for this model) and loads the `flash-next-experts.strp` profile, which keeps the most-used parts of the model on the GPU.

### The two fixes

**Fix 1: `GGML_MOE_CPU_SPLIT=0` (doubles writing speed)**

On this machine the settings file defaults to `GGML_MOE_CPU_SPLIT=1`, which controls how work is split between CPU and GPU. With it on, the short question ran at **~51 tok/s**. Turning it off gave **96.7 tok/s**.

Things I ruled out first: RAM (the disk stayed idle, and removing the 64 GB limit gave the same 47.6 tok/s) and thread count (16 threads instead of 30: 48.5 tok/s).

**Fix 2: `-ub 4096` (about 2.8× faster prompt reading)**

With the default batch size, a 100K prompt took **463.6 s** (221 tok/s). With `-ub 4096`: **165.9 s** (617 tok/s).

### Launch script used (`run-llamampere.sh`)

```bash
#!/usr/bin/env bash
# usage: run-llamampere.sh <context> [--exact] [extra flags, e.g. -ub 4096]
CTX=${1:-32768}; shift
export GGML_MOE_CPU_SPLIT=0
export LLAMA_SERVER_BIN=$HOME/llamAmpere/build-sm86/bin/llama-server
export FLASH_NEXT_DRAFTER=/models/drafter/mtp-Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2S-B.gguf
exec $HOME/llamAmpere/scripts/flash-next-server.sh "$@" \
  -m /models/swift-fn/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  -c $CTX --kv-gpu-tokens 32768 \
  --jinja --parallel 1 --cache-ram 0 -t ${T:-30} -td ${T:-30} --host 127.0.0.1 --port 8080
```

```bash
./run-llamampere.sh 131072 -ub 4096
```

### About `--exact`

By default the launcher uses a shortcut (`--moe-skip`) that skips some less important work. It's faster, but the output is not bit-for-bit the original model. Pass `--exact` to turn it off. **All results here use the launcher defaults.**

### Known limit: ~250K context on 24 GB

At ~250K tokens llamAmpere ran out of VRAM:

```
allocating 1764.17 MiB on device 0: cudaMalloc failed: out of memory
```

Lowering the batch to `-ub 2048` gave the same error.

---

## 3. Benchmark script

`bench.py` sends one streaming request to the OpenAI-compatible endpoint on `:8080` and records time to first token, reading speed and writing speed.

```bash
# python3 bench.py <engine-name> <approx_prompt_tokens> <max_tokens> [temp] [seed]
python3 bench.py strata-mtp 0 512                # short question
python3 bench.py strata-mtp 100000 512           # ~100K prompt
python3 bench.py strata-mtp 250000 512           # ~250K prompt
python3 bench.py strata-mtp-claim 100000 5120 1.0 23   # claim test
```

- Long prompts are filled with real C++ source code (llama.cpp), followed by "write a detailed review of this code".
- Each request starts with `[run <nanosecond timestamp>]`, so no prompt cache is reused.
- Results are appended to `results.jsonl`.

Update the path to the C++ source file at the top of the script to match your machine.

---

## 4. Raw results

Every run, including the failed and warm-up ones, is in [`results.jsonl`](results.jsonl). Full server logs for every llamAmpere run are in [`logs/`](logs/).

| File | What |
|---|---|
| `bench.py` | Benchmark script |
| `run-llamampere.sh` | llamAmpere launcher with both fixes |
| `results.jsonl` | Every run, both engines |
| `logs/llamampere-*.log` | llamAmpere server logs (32K, 128K, 256K, claim test, each fix) |

---

## Notes

- This was tested on llamAmpere **v0.6.2**. A point release (v0.6.2.1) came out on 4 Nov; results may differ.
- Numbers are from one machine. Your CPU, RAM speed and PCIe link will change them.
- Found something different on your card? Open an issue with your `results.jsonl` line and hardware.

**If this saved you time, a ⭐ on the repo and a sub on [Network Coder](https://www.youtube.com/@networkcoder) help a lot.**
