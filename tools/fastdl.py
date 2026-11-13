# -*- coding: utf-8 -*-
# aimp 多连接分段下载器 (标准库自研, 对外网限速源叠加带宽)
# 用法: python fastdl.py <url> <输出文件> [并发数=12]
import urllib.request, os, sys, threading, time

URL, OUT = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 12
PART = 8 * 1024 * 1024  # 每任务分片 8MB, 断点续传粒度

def opener():
    return urllib.request.build_opener()

def total_size():
    try:
        req = urllib.request.Request(URL, method='HEAD')
        with opener().open(req, timeout=30) as r:
            return int(r.headers['Content-Length'])
    except Exception:
        pass
    # 兜底: GET Range 0-0 解析 Content-Range
    req = urllib.request.Request(URL)
    req.add_header('Range', 'bytes=0-0')
    with opener().open(req, timeout=30) as r:
        cr = r.headers.get('Content-Range', '')
        return int(cr.split('/')[-1])

class Down:
    def __init__(self):
        self.lock = threading.Lock()
        self.done = 0
        self.t0 = time.time()

    def chunk(self, idx, start, end):
        path = '%s.p%04d' % (OUT, idx)
        want = end - start + 1
        for attempt in range(5):
            have = os.path.getsize(path) if os.path.exists(path) else 0
            if have > want:
                # 超写 = 旧 bug 残留 (重试时 Range 用旧 have 重复追加), 整片重下
                os.remove(path)
                have = 0
            if have >= want:
                return
            # 每次重试重建请求: Range 必须跟随真实 have, 禁止复用旧 Request
            req = urllib.request.Request(URL)
            req.add_header('Range', 'bytes=%d-%d' % (start + have, end))
            try:
                r = opener().open(req, timeout=60)
                if r.status != 206:
                    r.close()
                    time.sleep(2 * attempt + 1)
                    continue
                with r, open(path, 'ab') as f:
                    while True:
                        b = r.read(1 << 16)
                        if not b: break
                        f.write(b)
                        with self.lock:
                            self.done += len(b)
                if os.path.getsize(path) >= want:
                    return
            except Exception:
                time.sleep(2 * attempt + 1)
        raise IOError('chunk %d failed' % idx)

def main():
    lock = OUT + '.lock'
    if os.path.exists(lock):
        pid = open(lock).read().strip()
        print('another instance running (pid %s), exit' % pid)
        sys.exit(2)
    open(lock, 'w').write(str(os.getpid()))
    try:
        _download()
    finally:
        os.remove(lock)

def _download():
    size = total_size()
    print('total %.2f GB, %d connections' % (size / 1e9, N))
    # 任务队列: 每 PART 字节一个片, 轮转分配给 N 个线程
    ranges = [(s, min(s + PART, size) - 1) for s in range(0, size, PART)]
    down = Down()
    errs = []
    def worker(wid):
        for idx in range(wid, len(ranges), N):
            try:
                down.chunk(idx, ranges[idx][0], ranges[idx][1])
            except Exception as e:
                errs.append((idx, str(e)))
    ths = [threading.Thread(target=worker, args=(w,)) for w in range(N)]
    for t in ths: t.start()
    while any(t.is_alive() for t in ths):
        time.sleep(5)
        el = time.time() - down.t0
        print('\r%6.1f%%  %7.1f MB/s  eta %5.0fs ' % (
            down.done / size * 100, down.done / el / 1e6,
            (size - down.done) / max(down.done / el, 1)), end='', flush=True)
    for t in ths: t.join()
    if errs:
        print('\nFAILED chunks:', errs[:5]); sys.exit(1)
    print('\nmerge...')
    with open(OUT, 'wb') as f:
        for idx in range(len(ranges)):
            p = '%s.p%04d' % (OUT, idx)
            with open(p, 'rb') as pf:
                while True:
                    b = pf.read(1 << 22)
                    if not b: break
                    f.write(b)
            os.remove(p)
    print('DONE %s (%d bytes)' % (OUT, os.path.getsize(OUT)))

if __name__ == '__main__':
    main()
