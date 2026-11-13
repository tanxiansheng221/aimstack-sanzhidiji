# -*- coding: utf-8 -*-
# 模拟 aimw.c 装载器逐条校验, 定位 rc=2 的位置
import struct, sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
path = sys.argv[1] if len(sys.argv) > 1 else 'build/mini_v2.aimw'
buf = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', path), 'rb').read()
print('file size', len(buf))
pos = 0
assert buf[:4] == b'AIMW'
pos = 4
ver, nt = struct.unpack_from('<II', buf, pos); pos += 8
print('ver', ver, 'nt', nt)
for i in range(nt):
    (nl,) = struct.unpack_from('<I', buf, pos); pos += 4
    name = buf[pos:pos+nl].decode('utf-8'); pos += nl
    dt, nd = struct.unpack_from('<II', buf, pos); pos += 8
    dims = struct.unpack_from('<%dI' % nd, buf, pos); pos += 4*nd
    n = 1
    for d in dims: n *= d
    (nb,) = struct.unpack_from('<I', buf, pos); pos += 4
    if dt == 0: want = n*4
    elif dt == 1: want = n; 
    elif dt == 2: want = (n+3)//4
    elif dt == 3:
        if n % 256: print('REJECT n%256', name, n); break
        want = 32 + (n//256)*144
    else:
        print('REJECT dtype', name, dt); break
    ok = (nb == want) if dt in (0,3) else (nb == want + 4)
    print('%-28s dt=%d dims=%s n=%7d nb=%7d want=%7d %s' % (name, dt, dims, n, nb, want, 'OK' if ok else 'MISMATCH'))
    pos += nb
