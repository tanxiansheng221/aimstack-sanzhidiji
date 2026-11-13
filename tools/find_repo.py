# -*- coding: utf-8 -*-
# find MiniCPM5 GGUF repo on hf-mirror (restore lost download URL)
import json, urllib.request

def get(url, t=25):
    req = urllib.request.Request(url, headers={'User-Agent': 'aimp/0.1'})
    with urllib.request.urlopen(req, timeout=t) as r:
        return r.read().decode('utf-8', 'replace')

try:
    data = json.loads(get('https://hf-mirror.com/api/models?search=MiniCPM5&limit=30'))
    for m in data:
        print(m.get('id', '?'), '| likes', m.get('likes', 0), '| dl', m.get('downloads', 0))
except Exception as e:
    print('hf-mirror search FAIL:', e)

# for each candidate, list gguf files
try:
    for m in data[:15]:
        mid = m.get('id', '')
        try:
            info = json.loads(get('https://hf-mirror.com/api/models/' + mid + '?blobs=true'))
            for s in info.get('siblings', []):
                fn = s.get('rfilename', '')
                if fn.lower().endswith('.gguf'):
                    sz = s.get('size') or 0
                    print('  GGUF:', mid, fn, '%.2fGB' % (sz / 1e9))
        except Exception as e:
            print('  skip', mid, e)
except NameError:
    print('no candidates')
