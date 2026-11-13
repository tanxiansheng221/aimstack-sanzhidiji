# -*- coding: utf-8 -*-
# aimp M2.5: Python 参考 BPE (对账工具, 不进产物)
import struct, sys

def read_gguf_tokenizer(path):
    f = open(path, 'rb')
    def u32(): return struct.unpack('<I', f.read(4))[0]
    def u64(): return struct.unpack('<Q', f.read(8))[0]
    def rd(n): return f.read(n)
    def gs():
        n = u64()
        return rd(n).decode('utf-8', errors='replace')
    magic = struct.unpack('<I', rd(4))[0]
    assert magic == 0x46554747
    ver = u32()
    ntensor = u64()
    nkv = u64()
    _FMT = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'B',10:'Q',11:'q',12:'d'}
    _SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    def value(vt):
        if vt == 8: return gs()
        if vt == 9:
            et = u32(); cnt = u64()
            if et == 8: return [gs() for _ in range(cnt)]
            fmt = '<' + _FMT[et]*cnt
            return list(struct.unpack(fmt, rd(_SZ[et]*cnt)))
        return struct.unpack('<' + _FMT[vt], rd(_SZ[vt]))[0]
    kv = {}
    for _ in range(nkv):
        k = gs(); vt = u32()
        kv[k] = value(vt)
    return kv

if False:
    kv = read_gguf_tokenizer(sys.argv[1] if len(sys.argv) > 1
                             else 'data/m2b_head.gguf')
    toks = kv.get('tokenizer.ggml.tokens', [])
    merges = kv.get('tokenizer.ggml.merges', [])
    print('pre   =', kv.get('tokenizer.ggml.pre'))
    print('model =', kv.get('tokenizer.ggml.model'))
    print('tokens=', len(toks), ' merges=', len(merges))
    print('tok[0:6] =', toks[:6])
    print('merges[0:6] =', merges[:6])
    tt = kv.get('tokenizer.ggml.token_type', [])
    print('token_type len =', len(tt))
# ---- byte-level BPE ----
def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 257))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

B2U = bytes_to_unicode()
U2B = {v: k for k, v in B2U.items()}

class RefBPE:
    def __init__(self, kv):
        self.tokens = kv['tokenizer.ggml.tokens']
        self.tid = {t: i for i, t in enumerate(self.tokens)}
        self.ranks = {}
        for r, m in enumerate(kv.get('tokenizer.ggml.merges', [])):
            self.ranks[m] = r
        self.byte_id = {}
        for b, u in B2U.items():
            self.byte_id[b] = self.tid.get(u)

    def bpe(self, piece):
        if piece in self.tid:
            return [self.tid[piece]]
        ids = [self.tid[c] for c in piece]
        while len(ids) > 1:
            best, br = None, 1 << 30
            for i in range(len(ids) - 1):
                r = self.ranks.get(self.tokens[ids[i]] + ' ' + self.tokens[ids[i+1]])
                if r is not None and r < br:
                    br, best = r, i
            if best is None:
                break
            merged = self.tokens[ids[best]] + self.tokens[ids[best+1]]
            ids[best:best+2] = [self.tid[merged]]
        return ids

    def encode(self, text):
        """greedy segments: ' word' / word / number / ws / single-symbol"""
        ids = []
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            j = i
            if ch == ' ' and i + 1 < n and not text[i+1].isspace():
                j = i + 1
                if text[j].isalpha():
                    while j < n and text[j].isalpha(): j += 1
                elif text[j].isdigit():
                    while j < n and text[j].isdigit(): j += 1
                else:
                    j = i + 2
                seg = text[i:j]
            elif ch.isalpha():
                while j < n and text[j].isalpha(): j += 1
                seg = text[i:j]
            elif ch.isdigit():
                while j < n and text[j].isdigit(): j += 1
                seg = text[i:j]
            elif ch.isspace():
                j = i + 1
                seg = text[i:j]
            else:
                seg = ch; j = i + 1
            u = ''.join(B2U[b] for b in seg.encode('utf-8'))
            ids += self.bpe(u)
            i = j
        return ids

    def decode(self, ids):
        out = bytearray()
        for i in ids:
            t = self.tokens[i]
            ok = all(c in U2B for c in t)
            if ok:
                out += bytes(U2B[c] for c in t)
            else:
                out += t.encode('utf-8')
        return out.decode('utf-8', errors='replace')
# ---- self test: roundtrip + smoke ----
if __name__ == '__main__' and len(sys.argv) > 2:
    kv = read_gguf_tokenizer('data/m2b_head.gguf')
    bp = RefBPE(kv)
    mode = sys.argv[1]
    if mode == 'enc':
        print(' '.join(map(str, bp.encode(sys.argv[2]))))
    elif mode == 'dec':
        ids = [int(x) for x in sys.argv[2].split(',')]
        print(bp.decode(ids))
    elif mode == 'rt':
        ok = True
        for t in sys.argv[2:]:
            ids = bp.encode(t)
            back = bp.decode(ids)
            good = (back == t)
            ok = ok and good
            print('%s  %r -> %s -> %r' % ('PASS' if good else 'FAIL', t, ids[:12], back))
        print('==== RT_%s ====' % ('PASS' if ok else 'FAIL'))
