# turboquant-godot

Godot fork carrying `modules/llm` (in-process llama.cpp inference) and a
TurboQuant-modified `thirdparty/llama_cpp`.

Workspace-standing constraints live in `.repo/manifests/CLAUDE.md` and bind this
repo. This file records only what is specific to the fork. Where the two
disagree, the manifest wins.

## Commit convention

Sentence format, matching existing history: capitalized, imperative, terminal
period, no `type:` prefix.

```
Use clang sanitizers.
Install vulkan sdk.
Add LLM module with llama.cpp for on-device inference
```

Conventional-commit prefixes (`feat:`, `fix:`, `chore:`) are not used here.

## Vendored llama.cpp base

`thirdparty/llama_cpp` is a **git subtree**, converted from git-subrepo. There
is no metadata file: git subtree records the split point in the merge commit
message, and that message is the source of truth.

```
git-subtree-dir:   thirdparty/llama_cpp
git-subtree-split: 67559e580b10e4e47e9a6fd6218873997976886d
```

| field | value |
|---|---|
| remote (`turboquant`) | `https://github.com/TheTom/llama-cpp-turboquant` |
| branch | `feature/turboquant-kv-cache` |
| split commit | `67559e580b10e4e47e9a6fd6218873997976886d` |

That repo is public and is where TurboQuant is actually developed — the KV
cache work has real history there across many branches. **The rebase happens in
that repo, not in this one.** Reconstructing a delta from the squashed vendored
tree is the wrong approach; rebase `feature/turboquant-kv-cache` onto upstream
and then pull the result back:

```
git subtree pull --prefix=thirdparty/llama_cpp turboquant <branch> --squash
```

The remote was fetched with `--depth=1` at the split commit, so this repo did
not absorb the fork's full history.

### The local delta

The vendored tree is **not** a clean copy of the split commit. Godot-local
changes ride on top, and they were invisible under git-subrepo — a plain
subtree import reverts every one of them. They are now a single explicit
commit (`Reapply Godot-local llama.cpp changes on top of the subtree.`) so a
later `subtree pull` conflicts against something reviewable:

| path | change |
|---|---|
| `common/common.cpp` | macOS pre-10.15 filesystem fallback (`dirent`/`stat`) |
| `ggml/src/ggml-backend-dl.h`, `ggml-backend-reg.cpp` | `ggml-no-backend-dl.patch`, pre-applied |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | `ggml-vulkan-volk.patch`, pre-applied |
| `ggml/src/ggml-cpu/arch/x86/quants.c` | `_mm_prefetch` cast to `const char *` |
| `ggml/src/ggml-webgpu/ggml-wgsl-shaders.hpp` | generated, committed Godot-side only |
| ~10 paths | pruned by vendoring (android/swiftui examples, `.gen.h`, `build-xcframework.sh`) |

The patch files under `modules/llm/patches/` are **documentation of changes
already applied**, not something the build applies. Do not apply them again.

The conversion itself was content-neutral: the tree afterward is byte-identical
to the tree before, except `.gitrepo` is gone. That was verified by diffing
against a snapshot taken before the conversion, and it is the only guarantee
that the mechanism change did not quietly revert code.

`8ab23945b6` is the single commit that first landed the subrepo here, so this
repo's history shows no upstream detail before the conversion.

### Corroborating upstream base

Independently of `.gitrepo`, the vendored tree's upstream base was recovered by
blob-fingerprinting files TurboQuant does not touch against `ggml-org/llama.cpp`
history. This says where the TurboQuant fork branched from upstream, which is
what governs conflict surface:

| fact | value |
|---|---|
| upstream base commit | `ca7f7b7b9` |
| base date | 2026-04-21 |
| upstream remote | `https://github.com/ggml-org/llama.cpp` |

Method: for each clean file, find the newest upstream commit whose tree entry
hash equals `git hash-object` of the vendored copy; the base is the newest
commit where **all** clean files match simultaneously. Nine files agree on
`ca7f7b7b9`. Re-derive with `modules/llm/check_claims.py --base`.

The boundary is tight and easy to get wrong by one commit: the direct child
`134d6e54` ("common/chat, server: refactor…", #20690) rewrites
`common/chat.cpp`, so it fails the fingerprint. Deriving the base from a single
file's next-change date lands on `134d6e54` and is wrong — all nine files must
match at once. `check_claims.py --base` caught exactly this error.

Files that do **not** match upstream are fork-modified and are the rebase
conflict surface. `include/llama.h` and `src/llama-model-loader.cpp` are among
them — they are not clean, and using them as fingerprints yields a false
negative.

## TurboQuant fork surface

TurboQuant is not confined to `ggml/`. It adds three KV cache quantization types
in `ggml/include/ggml.h`:

| type | id | description |
|---|---|---|
| `GGML_TYPE_TURBO2_0` | 42 | WHT + 2-bit PolarQuant |
| `GGML_TYPE_TURBO3_0` | 43 | WHT + 3-bit PolarQuant |
| `GGML_TYPE_TURBO4_0` | 44 | WHT + 4-bit PolarQuant |

Exposed to GDScript as `LLMContext.cache_type_k` / `cache_type_v` values
`turbo2`, `turbo3`, `turbo4` (alongside `f16`, `q8_0`, `q4_0`).

It also modifies files inside upstream's `src/` and `common/`, which is the part
that makes a rebase a merge rather than a fast-forward:

```
src/llama-kv-cache.{h,cpp}   src/llama-graph.cpp      src/llama-context.cpp
src/llama-memory-hybrid.{h,cpp}  src/llama-memory.h   src/llama-model-loader.cpp
src/turbo-rotation-data{,-32}.h  include/llama.h      common/arg.cpp
tools/llama-bench/llama-bench.cpp   tools/server/…
```

Plus new standalone ggml files (low conflict risk): `ggml/src/ggml-turbo-quant.c`,
`ggml/src/ggml-cuda/turbo-{wht,innerq}.{cu,cuh}`, `ggml/src/ggml-cuda/mmvq-tq.cu`.

Local patches applied on top: `modules/llm/patches/ggml-no-backend-dl.patch`,
`modules/llm/patches/ggml-vulkan-volk.patch`.

## Qwen3.8 and MTP status

Qwen3.8 **loads today**. Its GGUF architecture is `qwen35`, and the vendored tree
has `LLM_ARCH_QWEN35` with a working `src/models/qwen35.cpp`. The size table at
`src/llama-model.cpp` already covers 0.8B/2B/4B/9B/27B.

What is missing is **MTP**, in four places:

1. `LLM_ARCH_QWEN35` never reads `LLM_KV_NEXTN_PREDICT_LAYERS`
2. it never creates `nextn.*` tensors, so the MTP head is never loaded
3. `src/models/qwen35.cpp` has no MTP graph
4. `common_speculative_type` has no `draft-mtp`; `--spec-type` offers only
   `none|ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod`

Upstream fix is `25558268` ("llama + spec: MTP Support", #22673), merged
2026-05-16, 54 files, +2226/-412. It is an ancestor of upstream `master`.

Separately, `LLMChat` runs a plain `llama_decode` loop and does not use
`common/speculative` at all. Landing MTP in llama.cpp does not by itself make
the module faster; wiring speculation into `llm_chat.cpp` is its own change.

## Test model

`cygnal/Qwen3.8-27B-heretic-ara-Q4_K_M-MTP-GGUF`, Apache-2.0, ungated.

| fact | value |
|---|---|
| bytes | 16810714560 |
| sha256 | `9d9b864f8a378721e9a78f87dec3161621217795843982d09764237ce7b86210` |
| local path | `~/models/qwen3.8-27b-mtp/Qwen3.8-27B-heretic-ara-Q4_K_M-MTP.gguf` |

There is no small Qwen3.8 — Qwen shipped only `Qwen3.8-27B` and
`Qwen3.8-2.4T-A95B`. Every Qwen3.5 size (0.8B/2B/4B/9B/27B) is the same
`qwen3_5` architecture with `mtp_num_hidden_layers = 1` and is Apache-2.0, so a
small Qwen3.5 exercises the identical `qwen35` MTP path when a fast loop is
wanted.

## Push authority

We push only to orgs we **own**. Being a member or a close collaborator is not
authority.

| org | rights |
|---|---|
| `V-Sekai-fire` | owned, push allowed |
| `v-sekai-multiplayer-fabric` | owned, push allowed |
| everything else (`godotengine`, `ggml-org`, `TheTom`, `sudoingX`, …) | fetch only |

Fetching from anyone is fine. The guard is mechanical rather than advisory:
every remote outside an owned org has its push URL set to
`DISABLED-not-our-org-fetch-only`, so a push fails instead of landing somewhere
we do not control. `check_claims.py` asserts this across this repo and every
sibling repo in the workspace, scratch clones included — that check found live
push URLs on two throwaway clones that were easy to forget.

Adding a remote is therefore two steps: `git remote add`, then
`git remote set-url --push <name> DISABLED-not-our-org-fetch-only` unless it is
an owned org.

### Consequence for the rebase

The TurboQuant rebase **cannot** be pushed to `TheTom/llama-cpp-turboquant`.
Fork or mirror that branch into `V-Sekai-fire` first, rebase there, and pull the
subtree from our own fork. The `turboquant` remote here stays fetch-only and
exists purely to read the fork's history.

## Where inference runs

`.repo/manifests/CLAUDE.md` states GPU work runs on RunPod, never on the local
desktop GPU. Benchmarking with `-ngl` on the Mac mini's Metal backend is local
GPU work and is **not permitted**. The rebase and conflict resolution are CPU
work and stay local; every build and every measurement runs on RunPod.

Pod: **RTX A6000 48GB**, $0.33/hr. Chosen because 48 GB clears the 262K window
and the full n-max 2-6 sweep without context pressure, and because `qwen38-mtp`
publishes an A6000 sweep to baseline against (26.7 spec-off, 52.5 at n-max 2,
peak 64.1 at n-max 4).

The manifest's teardown rule binds: tear the pod down after use, **double-check
the teardown**, and commit and push anything that matters before the machine
goes away. Nothing of value lives only on the pod.

RunPod credential is 1Password item `c76zprxgigzvawtfqxzgbsdyk4` ("Runpod.io API
Credentials", Personal vault, field `credential`). Read it at point of use with
`op read`; never write it to a file, a commit, or a log.

## Rebase plan (staged)

Gall's Law: the working complex system has to grow from a working simple one.
The rebase is the expensive, conflict-heavy step, so it does **not** go first.
Stage 0 carries no TurboQuant and no conflicts, and it retires the riskiest
unknown — whether this GGUF's MTP head survived a third-party requantization of
an abliterated derivative at all. If stage 0 fails, every later stage was
wasted motion.

| stage | tree | question it answers |
|---|---|---|
| 0 | stock upstream, unmodified | Does the head load, is `--spec-type draft-mtp` accepted, what is the spec-off floor, does `probe.py` run clean? |
| 1 | TurboQuant rebased onto `25558268` | Does TurboQuant survive the rebase, and does MTP still work beside turbo KV? |
| 2 | stage 1 rebased onto `master` | Does four more months of upstream change the result? |
| 3 | `LLMChat` wired to `common/speculative` | Does the Godot module see the gain? |

Each stage re-runs the same A/B, so any delta is attributable to one jump.
Stage 0 runs stock upstream: if it will not produce a gain there, no amount of
rebasing will produce one here.

Work happens in a clone of `TheTom/llama-cpp-turboquant`, rebasing
`feature/turboquant-kv-cache`. Only after a stage is green does it come back
here via `git subrepo pull`. Landing MTP in llama.cpp is necessary but not
sufficient: `LLMChat` runs a plain `llama_decode` loop and never calls
`common/speculative`, so stage 3 is what makes the module itself faster.

### Declared stage

`thirdparty/llama_cpp` currently has no MTP support. The marker below is
machine-read by `check_claims.py`, which fails if the tree and the declaration
disagree **in either direction**, or if a rebase lands only partially (nextn
tensors loaded but no graph, say). Update it in the same change that lands MTP.

<!-- gate:mtp-state=absent -->


Work happens in a clone of `TheTom/llama-cpp-turboquant`, rebasing
`feature/turboquant-kv-cache`. Only after a stage is green does it come back
here via `git subrepo pull`. Note that landing MTP in llama.cpp is necessary but
not sufficient: `LLMChat` must also be taught to use `common/speculative`, or
the module keeps its plain decode loop and sees none of the gain.

## Measuring MTP

The method comes from `qwen38-mtp` (Apache-2.0, docs only — no code to vendor).
Per the manifest's "a number without a baseline is not a measurement": every
reported figure is a paired A/B on one card, same GGUF and same config on both
sides, differing only in the spec flags, medians of 3 runs x 3 prompts, warmup
discarded. Report the spec-off floor in the same table as the spec-on number.

`qwen38-mtp/probe.py` is the streaming client that produces these numbers; it is
the negative control's reference implementation and should not be forked.

## Checking this file

`modules/llm/check_claims.py` verifies the falsifiable claims above (base
commit, ggml type ids, the four MTP gaps, model size and hash, upstream PR
metadata) and exits non-zero on drift. It ships a negative control:
`--self-test` asserts the checker fails on deliberately broken input.
