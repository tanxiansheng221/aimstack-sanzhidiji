# -*- coding: utf-8 -*-
# AIM-W 全文件校验器: 复刻 C 解析器全部拒绝条件, 报告第一个违规张量
import struct, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def validate(path):
    raw = open(path, 'rb').read()
    pos = 0
    assert raw[:4] == b'AIMW', 'magic'
    pos = 4
    ver, nt = struct.unpack_from('<II', raw, pos); pos += 8
    if ver not in (0, 1): return 'version %d' % ver
    for i in range(nt):
        nl = struct.unpack_from('<I', raw, pos)[0]; pos += 4
        name = raw[pos:pos+nl].decode('utf-8'); pos += nl
        dt, nd = struct.unpack_from('<II', raw, pos); pos += 8
        if nd == 0 or nd > 8: return 'tensor %d %s ndims=%d' % (i, name, nd)
        dims = struct.unpack_from('<%dI' % nd, raw, pos); pos += 4 * nd
        n = 1
        for d in dims: n *= d
        nb = struct.unpack_from('<I', raw, pos)[0]; pos += 4
        if dt == 0: want, tail = n * 4, 0
        elif dt == 1: want, tail = n, 4
        elif dt == 2: want, tail = (n + 3) // 4, 4
        else: return 'tensor %d %s dtype=%d' % (i, name, dt)
        if nb != want + tail:
            return 'tensor %d %s nb=%d want=%d' % (i, name, nb, want + tail)
        if pos + nb > len(raw):
            return 'tensor %d %s truncated' % (i, name)
        data = raw[pos:pos+want]
        if dt == 1:
            bad = [e for e in data if e not in (0, 1, 255)]
            if bad: return 'tensor %d %s bad int8 code %r' % (i, name, bad[:3])
        elif dt == 2:
            for b in range(want):
                byte = data[b]
                for s in range(4):
                    g = b * 4 + s
                    code = (byte >> (s * 2)) & 3
                    if g < n:
                        if code == 3: return 'tensor %d %s code3 @%d' % (i, name, g)
                    else:
                        if code != 0: return 'tensor %d %s padding @%d' % (i, name, g)
        pos += nb
    if pos != len(raw):
        return 'trailing %d bytes' % (len(raw) - pos)
    return 'CLEAN (ntensors=%d)' % nt

for p in sys.argv[1:]:
    print(p, '->', validate(p))
