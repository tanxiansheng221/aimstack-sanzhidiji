# -*- coding: utf-8 -*-
# run_local_rep.py — 本地过夜复现：748K 级 × 20K 题池 × 30K 步（DCU v5 正例配方，打包路径）
import os, sys
os.environ.update({
    "AIM_TAG": "local20k_rep", "AIM_OUT": r"H:\AimCode\lab\cpu-native\train_out",
    "AIM_CACHE": r"H:\AimCode\lab\cpu-native\local20k_pack_cache.npz", "AIM_PACK": "1",
    "AIM_D": "128", "AIM_L": "2", "AIM_H1": "512", "AIM_T": "512",
    "AIM_B": "4", "AIM_ACCUM": "4", "AIM_STEPS": "30000",
    "AIM_LR": "1e-3", "AIM_WARM": "500", "AIM_CHUNK": "64",
    "AIM_EVAL": "2000", "AIM_SAVE": "5000", "AIM_AMP": "off", "AIM_SEED": "42",
})
os.chdir(r"H:\AimCode\lab\cpu-native")
sys.argv = ["aim_train_real.py"]
import aim_train_real
aim_train_real.main()
