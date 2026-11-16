# Corpus Builder Manual

All corpus **generators** are open-source and reproducible; the datasets themselves are not committed (each has upstream licensing, see NOTICE.md) — scripts pull from official channels.

## Curriculum-type (ten subjects, deterministic generators)

| Generator | Subject |
|---|---|
| `curriculum_math_lab.py` | Math |
| `curriculum_mod_arith` (inside math) | Modular arithmetic |
| `curriculum_causal_lab.py` | Causal |
| `curriculum_csp_lab.py` | Constraint satisfaction |
| `curriculum_fsm_lab.py` | State machines |
| `curriculum_graph_lab.py` | Graph |
| `curriculum_logic_lab.py` | Logic |
| `curriculum_plan_lab.py` | Planning |
| `curriculum_progexec_lab.py` | Code execution |
| `curriculum_seqind_lab.py` | Sequence induction |

The full set follows five admission disciplines: "self-contained problem statements + targeted poisons must be caught + generator-embedded channel-B cross-check + degeneracy cap + multi-channel reconciliation" — generation is verification; every problem ships with a standard answer and an auditable chain.

Packing: `build_subject_curriculum.py` aggregates into training packs; `build_segment_caches.py` builds per-subject caches.

## Language / dialogue-type

| Script | Output | Upstream |
|---|---|---|
| `build_lang1_corpus.py` + `compress_lang1.py` | language base | Wikipedia + LCCC (`convert_lccc.py`) |
| `build_lang3b_corpus.py` / `build_lang3c_corpus.py` | Chinese dialogue | CPED (arXiv:2205.14727) |
| `build_lang4_slot.py` | slot arithmetic `[C:expr]=result` | self-produced (Calc-X style, fully evaluation-verified) |
| `build_lang5_refusal.py` / `build_lang5_pack.py` | Chinese refusal corpus (five categories: violence / porn / privacy / illegal / core assets) | self-produced synthesis |
| `fetch_cped.py` / `fetch_esconv.py` (tools legacy) | upstream fetch | ESConv (thu-coai) |

## Mixture pools

`build_v1_mix.py` / `build_804m_*.py` / `build_1p5b_*.py` / `build_pack_*.py` / `build_spec_pool.py` / `build_fmt*_corpus.py` (`fmt2_generators.py`/`fmt5_generators.py`): multi-source proportional mixing + packed caches (npz). Recipe details and per-version differences in `lab/docs/CORPUS_PLAN_LANG7.md`.

## Corpus autopsy (mandatory before training)

```bash
python lab/train/peek_lang_corpus.py <corpus file or directory>
```

Output: CJK byte ratio / ASCII ratio / language counts / window statistics. **Iron law**: small models get monolingual corpora; multilingual corpora are archived, never fed to small models (incident record in training.md).

## Upstream license quick reference

| Dataset | License essentials |
|---|---|
| Wikipedia dump | CC-BY-SA (attribution + share-alike) |
| CPED | academic use, see original repo |
| ESConv | original repo license |
| ROCStories / Opus-WritingPrompts | original distribution channel terms |
| prosocial-dialog | Apache-2.0 (original repo) |

When upstream licensing is unclear, skip rather than risk it (there are "honest skips" on record: OASST2 schema mismatch, some mirrors 401 — no brute-forcing).
