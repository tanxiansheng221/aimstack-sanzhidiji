# The CPU-Native Research Log — English Overview & Chapter Map

**Original**: [CPU_NATIVE_DEDUCTION.md](CPU_NATIVE_DEDUCTION.md) — 117 chapters, ~300k characters, in Chinese. This page is the entry guide for English readers: what the log covers, what it proved, and which chapters matter for which question.

## What this document is

A complete first-person build journal of the entire stack, written as it happened: every design decision, every measured number, every failed idea. It starts from "can a 2018 dual-core laptop do local LLM inference at all" and ends with a working ternary engine, a bit-exact reconciliation discipline, and a full archive of negative results. Nothing in the repo is undocumented — if a number appears in the README, the chapter where it was measured exists here.

## What it proved (headline results)

| Claim | Where |
|---|---|
| Ternary SIMD kernel 0.9 → 3.20 tok/s on BitNet b1.58-2B (2 cores) | §80.6 latency-wall teardown → multithreading → symbol-table decoding → k=2 fused speculative cores |
| Ternary SIMD fast path 85.53 tok/s (41× over baseline) | §80.6, with C/NumPy bit-exact reconciliation kept green through every rewrite |
| HF greedy decode bit-identical (first 9 tokens, token-for-token) | §80.6 reconciliation campaign |
| Speculative decoding: naive k=2 costs +21%; fused cores bring it to break-even on compute-bound hardware; k=4 blows the register file | §80.6.7 fused-core campaign |
| Continuation training from a converged checkpoint: three recipes all negative returns | §80.4 recipe campaigns |
| Ten-arm arithmetic comparison: zero real generalization (incl. a false-positive "0.232 score" that was an always-answer-0 prior) | E6 dossier |
| Corpus autopsy iron law: a six-language empathy corpus at 5% CJK bytes kept loss at the 2.0 plateau for 40k steps | language-first doctrine |

## Chapter map (where to look for what)

| If you want to know… | Read |
|---|---|
| Why pure C, why from scratch, what "supply-chain immunity" means in practice | the charter chapters (§80, "断供免疫宪章") |
| How the SIMD ternary kernel was brought up and validated bit-exactly | §80.6 series |
| How speculative decoding was made break-even (fused cores, register budget) | §80.6.7 and its supplements |
| Why continuation training kept failing (replay ratios, language-first) | §80.4 + language-doctrine chapters |
| What was tried and failed on arithmetic generalization (ten arms) | E6_CAMPAIGN_AND_RECIPE_DOSSIER.md |
| How the corpus builders and their five admission disciplines came to be | corpus-doctrine chapters |

## Why it ships in Chinese

The log was written in the language of the build itself, chapter by chapter, in real time — translating it after the fact would produce a second-hand text. This overview is the English entry point; the numbers, tables, and code snippets are language-neutral. If you read Chinese, the original is the primary source; if not, the README, the manuals, and this page carry everything decision-relevant.

## License

The research log is CC-BY-NC 4.0 (attribution, non-commercial) — see NOTICE.md.
