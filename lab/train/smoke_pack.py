# -*- coding: utf-8 -*-
# smoke_pack.py — 打包/非打包双路径本地 CPU 冒烟：可跑性 + loss 量级 + val 路径
import io, os, subprocess, sys, time

ROOT = r"H:\AimCode\lab\cpu-native"
TMP = os.path.join(ROOT, "_pack_smoke")
os.makedirs(TMP, exist_ok=True)

# 小混料：v3k2 头 4000 行
mix = os.path.join(TMP, "mix_head.jsonl")
if not os.path.exists(mix):
    with io.open(ROOT + r"\v3k2_804m_mix.jsonl", encoding="utf-8") as f, \
         io.open(mix, "w", encoding="utf-8", newline="") as g:
        for i, line in enumerate(f):
            if i >= 4000:
                break
            g.write(line)

base_env = dict(os.environ)
base_env.update({
    "AIM_MIX": mix, "AIM_OUT": os.path.join(TMP, "out"),
    "AIM_D": "96", "AIM_L": "2", "AIM_H1": "256", "AIM_T": "512",
    "AIM_B": "8", "AIM_ACCUM": "2", "AIM_STEPS": "100",
    "AIM_LR": "1e-3", "AIM_WARM": "10", "AIM_CHUNK": "64",
    "AIM_EVAL": "100", "AIM_SAVE": "100000", "AIM_AMP": "off", "AIM_SEED": "42",
})

for mode, pack in [("unpacked", "0"), ("packed", "1")]:
    env = dict(base_env)
    env["AIM_PACK"] = pack
    env["AIM_TAG"] = "smoke_" + mode
    env["AIM_CACHE"] = os.path.join(TMP, "cache_%s.npz" % mode)
    if os.path.exists(env["AIM_CACHE"]):
        os.remove(env["AIM_CACHE"])
    print("=" * 20, mode, "=" * 20, flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(ROOT, "aim_train_real.py")], env=env)
    print("%s exit=%d wall=%.0fs" % (mode, r.returncode, time.time() - t0), flush=True)
