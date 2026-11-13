# -*- coding: utf-8 -*-
# llama3 BPE 最小编解码 (纯 stdlib, 读 HF tokenizer.json)
# 用法: python llama3_tok.py encode "text" / decode "id,id,id"
import json, os, re, sys, unicodedata

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
TJ = os.path.join(HERE, '..', 'data', 'bitnet2b', 'tokenizer.json')

# GPT-2 byte<->unicode 表 (ByteLevel)
def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1)) + \
         list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

B2U = bytes_to_unicode()
U2B = {v: k for k, v in B2U.items()}

class Tok:
    def __init__(self, path=TJ):
        d = json.load(open(path, encoding='utf-8'))
        m = d['model']
        self.vocab = {t: i for t, i in m['vocab'].items()}
        self.ranks = {}
        for i, mg in enumerate(m['merges']):
            a, b = (mg.split(' ', 1) if isinstance(mg, str) else mg)
            self.ranks[(a, b)] = i
        self.rx = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+".replace('\\p{L}', '[^\\W\\d_]').replace('\\p{N}', '[0-9]'))
        self.added = {t['content']: t['id'] for t in d['added_tokens']}
        self.bos = self.added.get('<|begin_of_text|>', 128000)
        self.eos = self.added.get('<|end_of_text|>', 128001)
        # 处理 \p{L} 替换失败的情形: 用 unicodedata 兜底方案
        self.rx = None  # 统一走 _split

    def _split(self, text):
        # llama3 预分词正则的 python 实现 (unicodedata 版)
        pat = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w]?\w+|\d{1,3}| ?[^\s\w]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+", re.UNICODE)
        return pat.findall(text)

    def _bpe(self, word):
        parts = list(word)
        while len(parts) > 1:
            best, bi = None, None
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if bi is None:
                break
            parts[bi:bi + 2] = [parts[bi] + parts[bi + 1]]
        return parts

    def encode(self, text, add_bos=True):
        ids = [self.bos] if add_bos else []
        for w in self._split(text):
            piece = ''.join(B2U[b] for b in w.encode('utf-8'))
            for p in self._bpe(piece):
                if p in self.vocab:
                    ids.append(self.vocab[p])
                else:
                    for b in p.encode('utf-8'):  # 兜底: 逐字节
                        pass
                    for ch in p:
                        ids.append(self.vocab[B2U[U2B[ch]]])
        return ids

    def decode(self, ids):
        vocab_inv = {i: t for t, i in self.vocab.items()}
        chars = ''.join(vocab_inv[i] for i in ids if i in vocab_inv)
        out = bytearray()
        for ch in chars:
            if ch in U2B:
                out.append(U2B[ch])
            else:
                out.extend(ch.encode('utf-8'))
        return out.decode('utf-8', errors='replace')

if __name__ == '__main__':
    t = Tok()
    mode = sys.argv[1]
    if mode == 'encode':
        ids = t.encode(sys.argv[2])
        print('ENC_IDS:' + ' '.join(map(str, ids)))
    else:
        ids = [int(x) for x in sys.argv[2].split(',')]
        print('DEC_TEXT:' + t.decode(ids))
