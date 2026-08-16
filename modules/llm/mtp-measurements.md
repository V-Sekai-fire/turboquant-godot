# MTP measurements

Paired A/B, one card, same GGUF and same server config on both arms, differing
only in the spec flags. Medians of 3 runs x 3 prompts, warmup discarded,
thinking off, produced by `qwen38-mtp/probe.py` unmodified
(`sha256:d9ea0c9fb1043593…`). The spec-off floor is reported in the same table
as the spec-on number, because a number without a baseline is not a
measurement.

## Stage 0 — stock upstream, no TurboQuant

Answers only: does the MTP head load, is `--spec-type draft-mtp` accepted, and
what does it do. Carries no TurboQuant, so nothing here is attributable to the
fork.

| fact | value |
|---|---|
| tree | stock `ggml-org/llama.cpp` at `25558268` (the MTP merge, #22673) |
| GPU | RTX A6000 48GB, driver 580.159.03, CUDA 12.4, `sm_86` |
| host | RunPod, `$0.53/hr` as configured (80 GB container disk, 96 vCPU) |
| model | `cygnal/Qwen3.8-27B-heretic-ara-Q4_K_M-MTP-GGUF`, Apache-2.0 |
| model sha256 | `9d9b864f8a378721e9a78f87dec3161621217795843982d09764237ce7b86210` |
| server | `-c 131072 -ngl 999 -fa 1 --cache-type-k q4_0 --cache-type-v q4_0 --parallel 1` |
| MTP arm adds | `--spec-type draft-mtp --spec-draft-n-max 2` |

`--parallel 1` is set on **both** arms, so the spec flags are the only
difference. Context is 131072 against a 262144 train length, so the full window
is deliberately not used.

### Results (tok/s, decode)

| arm | overall median | P1 code (py) | P2 prose (mmap) | P3 code (bash) | acceptance |
|---|---|---|---|---|---|
| spec off (floor) | **34.4** | 34.5 | 34.5 | 34.3 | — |
| draft-mtp, n-max 2 | **57.0** | 62.0 | 46.5 | 57.0 | 0.56–0.84 |
| gain | **+65.7%** | +79.7% | +34.8% | +66.2% | |

Baseline run-to-run spread was 34.3–34.6 tok/s across all nine runs, so the
floor is tight and the gain is far outside the noise band.

The per-prompt shape matches every row in the `qwen38-mtp` table: code gains
most, prose least. Prose (P2) is the weakest arm at +34.8% while the Python
prompt is the strongest at +79.7%.

### Negative control

A silently-ignored flag looks exactly like "no speedup", and a misattributed
gain looks exactly like a real one, so the arms are checked for drafting
evidence rather than trusted:

| arm | drafting evidence in server log |
|---|---|
| spec off | **0 matches** for acceptance/draft counters |
| draft-mtp | `statistics draft-mtp`, 921 accepted / 1050 drafts, 1689 accepted tokens |

The MTP head therefore actually drafted, and the baseline actually did not. The
gain is attributable to the flag.

### What this does not show

- Nothing about TurboQuant. Stage 0 is stock upstream; turbo KV cache types are
  not in this tree.
- Nothing about the Godot module. `LLMChat` runs a plain `llama_decode` loop and
  never calls `common/speculative`, so `modules/llm` is unaffected by this
  result until that wiring lands.
- Not comparable line-for-line to the published A6000 row (26.7 → 52.5), which
  used Q8_K_XL at 256K context with q8_0 KV. Different quant, context and cache
  type; only the shape and the acceptance band are comparable.

## Stage 1 — TurboQuant rebased onto 25558268

Not yet run.

## Stage 2 — rebased onto upstream master

Not yet run.

## Stage 3 — LLMChat wired to common/speculative

Not yet run. This is the stage that makes `modules/llm` itself faster.
