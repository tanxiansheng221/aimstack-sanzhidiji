# NOTICE / Third-party content attribution

Third-party content contained in or referenced by this repository, listed by source:

## 1. Microsoft BitNet weights (MIT, not distributed — download and convert yourself)

- Model: [microsoft/bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
- Copyright: © Microsoft Corporation, **MIT license**
- Relationship to this repo: this repo only provides the **conversion tools** for the AIM-W (`.aimw`) packing format (`tools/bitnet2aimw.py`, `tools/gguf2aimw.py`) and the inference kernel implementation. Download the weights from Microsoft's official channel (hf-mirror.com works in China); this repo does not distribute or modify the weight values.
- "BitNet" and "b1.58" are names of Microsoft's research; references in this document are for technical description only.

## 2. Chinese Wikipedia content (self-built store, from dumps)

- Source: Chinese Wikipedia dump (Wikimedia Downloads / mirrors)
- License: **CC-BY-SA 3.0 / 4.0** (Wikipedia text license)
- Attribution obligation: redistributing the self-built store requires the same CC-BY-SA license, preserved attribution, and links to the original text (the pipeline keeps the original link in each card's `source` field for traceability). This repo does not distribute the store itself; users build it under the terms above.

## 3. Corpus upstream datasets (fetched via scripts, not redistributed)

The following datasets are pulled from official channels by fetch scripts in `lab/train/`; this repo does not redistribute their contents — comply with each license before use:

| Dataset | Source | Use |
|---|---|---|
| CPED | scut-bds/CPED (arXiv:2205.14727) | Chinese multi-turn dialogue |
| ESConv | thu-coai/Emotional-Support-Conversation | emotional support dialogue |
| PsyQA | volunteer public data | Chinese psychological QA |
| ROCStories / Opus-WritingPrompts | official releases | creative writing |
| prosocial-dialog | allenai | cautious response / intervention corpus |
| LCCC | thu-coai | Chinese dialogue base |
| Chinese Wikipedia dump | Wikimedia Downloads | knowledge base base |

## 4. Inference kernel reference

The ternary inference kernel's implementation references the 1.58-bit ternary weight computation method as publicly described in the BitNet b1.58 paper (Frostig et al., Microsoft Research). The code is an independent implementation with no Microsoft code copied.

## 5. Render cascade

Edge rendering uses the system-installed Microsoft Edge browser (Playwright `channel=msedge`); no browser binaries are distributed with this repo.
