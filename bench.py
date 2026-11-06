import json, time, sys, urllib.request
# usage: python3 bench.py <engine-name> <approx_prompt_tokens> <max_tokens> [temp] [seed]
eng, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
temp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 23
URL = "http://127.0.0.1:8080/v1/chat/completions"
chunk = open("/root/llamAmpere/src/llama.cpp").read()
text = (chunk * (n * 4 // len(chunk) + 1))[: n * 4]
txt = ("Here is some C++ code:\n" + text + "\n\nWrite a detailed review of this code.") if n > 200 else "Write a detailed explanation of how a hash map works, with a C++ example."
msg = f"[run {time.time_ns()}]\n" + txt
body = json.dumps({"model": "x", "messages": [{"role": "user", "content": msg}], "max_tokens": out,
                   "temperature": temp, "seed": seed, "stream": True,
                   "stream_options": {"include_usage": True}}).encode()
req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
t0 = time.time(); first = None; usage = {}; ntok = 0
with urllib.request.urlopen(req, timeout=7200) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"): continue
        d = json.loads(line[5:])
        if d.get("usage"): usage = d["usage"]
        for c in d.get("choices", []):
            dl = c.get("delta", {})
            if dl.get("content") or dl.get("reasoning_content"):
                ntok += 1
                if first is None: first = time.time()
t1 = time.time()
pt = usage.get("prompt_tokens", n); ct = usage.get("completion_tokens", ntok)
res = dict(engine=eng, prompt_tokens=pt, output_tokens=ct, ttft_s=round(first - t0, 1),
           prefill_tps=round(pt / (first - t0), 1), decode_tps=round((ct - 1) / (t1 - first), 1), temp=temp)
print(json.dumps(res)); open("/root/bench/results.jsonl", "a").write(json.dumps(res) + "\n")
