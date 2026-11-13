# -*- coding: utf-8 -*-
# ddp_launch.py — 手写 DDP 启动器（绕开 torchrun libuv 坑；DCU 上同理用 bash 多进程）
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def worker(rank):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = os.environ.get("AIM_WORLD", "2")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = os.environ.get("AIM_MASTER_PORT", "29777")
    os.environ["LOCAL_RANK"] = str(rank)
    import aim_train_real
    aim_train_real.main()

if __name__ == "__main__":
    import multiprocessing as mp
    world = int(os.environ.get("AIM_WORLD", "2"))
    procs = []
    ctx = mp.get_context("spawn")
    for r in range(world):
        p = ctx.Process(target=worker, args=(r,))
        p.start()
        procs.append(p)
    codes = []
    for p in procs:
        p.join()
        codes.append(p.exitcode)
    print("worker exitcodes:", codes)
    sys.exit(0 if all(c == 0 for c in codes) else 1)
