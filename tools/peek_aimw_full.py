# -*- coding: utf-8 -*-
# peek AIM-W header: tensor list, dtypes, dims (spec-exact, stdlib only)
import struct, sys, io
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def peek(path, limit=20):
    raw = open(path, 'rb').read(4096)
    pos = 0
    assert raw[pos:pos+4] == b'AIMW', 'bad magic'
    pos += 4
    ver, nt = struct.unpack_from('<II', raw, pos); pos += 8
    print(path)
    print('  version=%d ntensors=%d' % (ver, nt))
    import os
    f = open(path, 'rb'); full = f.read(); f.close()
    for i in range(nt):
        nl = struct.unpack_from('<I', full, pos)[0]; pos += 4
        name = full[pos:pos+nl].decode('utf-8'); pos += nl
        dt, nd = struct.unpack_from('<II', full, pos); pos += 8
        dims = struct.unpack_from('<%dI' % nd, full, pos); pos += 4 * nd
        nb = struct.unpack_from('<I', full, pos)[0]; pos += 4
        n = 1
        for d in dims: n *= d
        extra = ''
        if dt in (1, 2):
            sc = struct.unpack_from('<f', full, pos + nb - 4)[0]
            extra = ' scale=%.4g' % sc
        print('  [%2d] %-28s dt=%d dims=%s n=%d %dKB%s' % (i, name, dt, dims, n, nb // 1024, extra))
        pos += nb
        if i >= limit - 1 and nt > limit:
            print('  ... (%d more, skipping header walk)' % (nt - limit))
            break
    print('  total walk pos=%d filesize=%d' % (pos, len(full)))

for p in sys.argv[1:]:
    try:
        peek(p)
    except Exception as e:
        print(p, 'ERR', e)
