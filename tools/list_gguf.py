# -*- coding: utf-8 -*-
# list GGUF files of openbmb/MiniCPM5-2B-GGUF (hf-mirror API, 429-aware)
import json, time, urllib.request

def get(url, t=25, tries=6):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'aimp/0.1'})
            with urllib.request.urlopen(req, timeout=t) as r:
                return r.read().decode('utf-8', 'replace')
        except Exception as e:
            print('retry', a, e)
            time.sleep(8 * (a + 1))
    raise IOError('give up')

info = json.loads(get('https://hf-mirror.com/api/models/openbmb/MiniCPM5-2B-GGUF?blobs=true'))
for s in info.get('siblings', []):
    fn = s.get('rfilename', '')
    sz = s.get('size') or 0
    print(fn, '%.3fGB' % (sz / 1e9) if sz else '')
