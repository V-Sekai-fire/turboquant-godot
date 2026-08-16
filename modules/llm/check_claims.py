#!/usr/bin/env python3
"""Verify the falsifiable claims in turboquant-godot/CLAUDE.md.

Exits non-zero on drift. Unchecked items are named and counted rather than
omitted, because a silent skip reads exactly like a pass.

    python3 modules/llm/check_claims.py                 # structural claims
    python3 modules/llm/check_claims.py --hash          # also hash the 16.8 GB model
    python3 modules/llm/check_claims.py --base UPSTREAM # re-derive the vendored base commit
    python3 modules/llm/check_claims.py --self-test     # negative control

The negative control is not optional decoration: a gate that has never been
shown to fail is certifying nothing.
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LC = os.path.join(REPO, "thirdparty", "llama_cpp")

BASE_COMMIT = "ca7f7b7b9"
MODEL_BYTES = 16810714560
MODEL_SHA256 = "9d9b864f8a378721e9a78f87dec3161621217795843982d09764237ce7b86210"
MODEL_PATH = os.path.expanduser(
    "~/models/qwen3.8-27b-mtp/Qwen3.8-27B-heretic-ara-Q4_K_M-MTP.gguf"
)
MTP_UPSTREAM = "25558268"

# Files TurboQuant does not touch, used to fingerprint the vendored base.
# include/llama.h and src/llama-model-loader.cpp are deliberately absent: they
# ARE fork-modified, and including them yields a false negative.
CLEAN_FILES = [
    "src/llama-arch.cpp",
    "src/models/qwen35.cpp",
    "common/speculative.cpp",
    "common/common.h",
    "common/chat.cpp",
    "common/sampling.cpp",
    "src/llama-vocab.cpp",
    "src/llama-batch.cpp",
    "src/llama-hparams.cpp",
]

TURBO_TYPES = [("GGML_TYPE_TURBO2_0", 42), ("GGML_TYPE_TURBO3_0", 43), ("GGML_TYPE_TURBO4_0", 44)]


def arch_blocks(text, arch):
    """Every `case LLM_ARCH_<arch>:` block, up to the next case label."""
    out = []
    for m in re.finditer(r"case\s+LLM_ARCH_%s\s*:" % re.escape(arch), text):
        nxt = re.search(r"case\s+LLM_ARCH_\w+\s*:", text[m.end():])
        out.append(text[m.end(): m.end() + (nxt.start() if nxt else 4000)])
    return out


# --- checks: each takes content, returns (ok, detail) so the negative control
# --- can feed them deliberately broken input.

def check_turbo_types(ggml_h):
    missing = [n for n, i in TURBO_TYPES if not re.search(r"%s\s*=\s*%d" % (n, i), ggml_h)]
    return (not missing, "all three present" if not missing else "missing/renumbered: %s" % ", ".join(missing))


def check_gap_hparams(model_cpp):
    """Gap 1: QWEN35 must not read the nextn hparam (claim: MTP absent)."""
    blocks = arch_blocks(model_cpp, "QWEN35")
    if not blocks:
        return False, "no LLM_ARCH_QWEN35 case found -- arch missing entirely"
    hit = any("LLM_KV_NEXTN_PREDICT_LAYERS" in b for b in blocks)
    return (not hit, "absent as claimed" if not hit else "PRESENT -- MTP hparam has landed, CLAUDE.md is stale")


def check_gap_tensors(model_cpp):
    """Gap 2: QWEN35 must not create nextn.* tensors."""
    blocks = arch_blocks(model_cpp, "QWEN35")
    if not blocks:
        return False, "no LLM_ARCH_QWEN35 case found"
    hit = any(re.search(r"LLM_TENSOR_NEXTN|layer\.nextn", b) for b in blocks)
    return (not hit, "absent as claimed" if not hit else "PRESENT -- nextn tensors now loaded, CLAUDE.md is stale")


def check_gap_graph(qwen35_cpp):
    """Gap 3: qwen35.cpp must have no MTP graph."""
    hit = re.search(r"nextn|\bmtp\b|MTP", qwen35_cpp, re.IGNORECASE)
    return (not hit, "absent as claimed" if not hit else "PRESENT -- MTP graph has landed, CLAUDE.md is stale")


def check_gap_spectype(common_h, arg_cpp):
    """Gap 4: no draft-mtp speculative type or flag value."""
    in_enum = "DRAFT_MTP" in common_h or "COMMON_SPECULATIVE_TYPE_MTP" in common_h
    in_flag = "draft-mtp" in arg_cpp
    hit = in_enum or in_flag
    where = ", ".join([w for w, h in (("common.h enum", in_enum), ("arg.cpp flag", in_flag)) if h])
    return (not hit, "absent as claimed" if not hit else "PRESENT in %s -- CLAUDE.md is stale" % where)


def check_model_size(size):
    return size == MODEL_BYTES, "%d bytes" % size if size == MODEL_BYTES else "%d bytes, expected %d" % (size, MODEL_BYTES)


def check_model_hash(digest):
    return digest == MODEL_SHA256, digest[:16] + "..." if digest == MODEL_SHA256 else "got %s..." % digest[:16]


def check_base(upstream, vendored, expected=BASE_COMMIT):
    """Re-derive the base by blob-fingerprinting clean files against upstream."""
    def git(*a):
        return subprocess.run(["git", "-C", upstream] + list(a), capture_output=True, text=True).stdout.strip()

    if not git("rev-parse", "--git-dir"):
        return False, "%s is not a git repository" % upstream
    mismatched = []
    for f in CLEAN_FILES:
        p = os.path.join(vendored, f)
        if not os.path.exists(p):
            return False, "vendored file missing: %s" % f
        local = subprocess.run(["git", "hash-object", p], capture_output=True, text=True).stdout.strip()
        if git("rev-parse", "%s:%s" % (expected, f)) != local:
            mismatched.append(f)
    if mismatched:
        return False, "%d/%d clean files differ at %s: %s" % (
            len(mismatched), len(CLEAN_FILES), expected, ", ".join(mismatched[:3]))
    return True, "all %d clean files match %s" % (len(CLEAN_FILES), expected)


NEXTN_SUFFIXES = ["eh_proj", "enorm", "hnorm", "shared_head_norm"]
BLOCK_TENSORS = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_down", "ffn_gate", "ffn_up"]


def check_mtp_head(arch, nextn_layers, block_count, names):
    """The GGUF must carry a complete MTP block, not just the nextn projections.

    A third-party requant can keep the four nextn.* tensors and drop the MTP
    layer's own attention/FFN weights, which loads fine and then produces a head
    that cannot draft. Checking only for 'nextn' would pass that file.
    """
    if arch != "qwen35":
        return False, "architecture is %r, expected qwen35" % arch
    if not nextn_layers or nextn_layers < 1:
        return False, "nextn_predict_layers = %r" % nextn_layers
    if not block_count:
        return False, "no block_count in metadata"

    idx = block_count - 1  # MTP layer is the last block
    have = set(names)
    missing_nextn = [s for s in NEXTN_SUFFIXES if "blk.%d.nextn.%s.weight" % (idx, s) not in have]
    if missing_nextn:
        return False, "blk.%d missing nextn: %s" % (idx, ", ".join(missing_nextn))
    missing_block = [s for s in BLOCK_TENSORS if "blk.%d.%s.weight" % (idx, s) not in have]
    if missing_block:
        return False, "blk.%d has nextn but is not a full block, missing: %s" % (
            idx, ", ".join(missing_block))
    return True, "blk.%d complete: %d nextn + %d block tensors" % (
        idx, len(NEXTN_SUFFIXES), len(BLOCK_TENSORS))


def read_gguf_mtp(path, gguf_py):
    """Extract (arch, nextn_layers, block_count, tensor names) from a GGUF."""
    sys.path.insert(0, gguf_py)
    from gguf import GGUFReader

    r = GGUFReader(path, "r")

    def field(k):
        f = r.fields.get(k)
        if f is None:
            return None
        v = f.contents()
        return v.decode() if isinstance(v, bytes) else v

    return (field("general.architecture"), field("qwen35.nextn_predict_layers"),
            field("qwen35.block_count"), [t.name for t in r.tensors])


def declared_mtp_state(claude_text):
    """The stage CLAUDE.md declares. Machine-read so prose cannot drift from it."""
    m = re.search(r"gate:mtp-state=(absent|present)", claude_text)
    return m.group(1) if m else None


def check_mtp_state(model_cpp, qwen35_cpp, common_h, arg_cpp, declared):
    """Observed MTP support must match the declared stage, and must be all-or-nothing.

    This replaces four independent 'gap is absent' assertions, which would have
    flipped to failure the moment the rebase succeeded, and which could not see
    a half-landed rebase (tensors loaded but no graph) at all.
    """
    if declared not in ("absent", "present"):
        return False, "CLAUDE.md declares no gate:mtp-state=absent|present marker"

    blocks = arch_blocks(model_cpp, "QWEN35")
    if not blocks:
        return False, "no LLM_ARCH_QWEN35 case found -- arch missing entirely"

    obs = {
        "hparam": any("LLM_KV_NEXTN_PREDICT_LAYERS" in b for b in blocks),
        "tensors": any(re.search(r"LLM_TENSOR_NEXTN|layer\.nextn", b) for b in blocks),
        "graph": bool(re.search(r"nextn|\bmtp\b", qwen35_cpp, re.IGNORECASE)),
        "spec": ("DRAFT_MTP" in common_h or "COMMON_SPECULATIVE_TYPE_MTP" in common_h
                 or "draft-mtp" in arg_cpp),
    }
    present = [k for k, v in obs.items() if v]
    absent = [k for k, v in obs.items() if not v]

    if present and absent:
        return False, "PARTIAL rebase -- present: %s / absent: %s" % (
            ",".join(sorted(present)), ",".join(sorted(absent)))
    actual = "present" if present else "absent"
    if actual != declared:
        return False, "tree is %s but CLAUDE.md declares %s" % (actual, declared)
    return True, "all four %s, matches declared stage" % actual


def parse_gitrepo(text):
    """Parse the [subrepo] section of a .gitrepo file."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(";") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def check_subrepo_doc(gitrepo_text, claude_text):
    """CLAUDE.md must restate .gitrepo exactly. The artefact wins; prose drifts."""
    g = parse_gitrepo(gitrepo_text)
    need = ["remote", "branch", "commit"]
    missing_field = [k for k in need if not g.get(k)]
    if missing_field:
        return False, ".gitrepo has no %s" % ", ".join(missing_field)
    drifted = [k for k in need if g[k] not in claude_text]
    if drifted:
        return False, "CLAUDE.md disagrees with .gitrepo on: %s (artefact says %s)" % (
            ", ".join(drifted), ", ".join(g[k] for k in drifted))
    return True, "remote/branch/commit match .gitrepo"


def check_mtp_upstream(upstream, commit=MTP_UPSTREAM):
    """The MTP fix exists upstream and is an ancestor of master."""
    def git(*a):
        r = subprocess.run(["git", "-C", upstream] + list(a), capture_output=True, text=True)
        return r.returncode, r.stdout.strip()

    rc, subj = git("log", "-1", "--format=%s", commit)
    if rc != 0:
        return False, "commit %s not found in %s" % (commit, upstream)
    if "MTP" not in subj:
        return False, "%s is not the MTP commit: %r" % (commit, subj[:60])
    rc, _ = git("merge-base", "--is-ancestor", commit, "master")
    if rc != 0:
        return False, "%s is NOT an ancestor of master" % commit
    return True, "%s ancestor of master: %s" % (commit, subj[:44])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(16 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def self_test():
    """Negative control: every check must fail on deliberately broken input."""
    cases = [
        ("turbo types", lambda: check_turbo_types("GGML_TYPE_TURBO2_0 = 99,")),
        ("gap: hparams", lambda: check_gap_hparams(
            "case LLM_ARCH_QWEN35: { ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, x); } case LLM_ARCH_FOO:")),
        ("gap: tensors", lambda: check_gap_tensors(
            "case LLM_ARCH_QWEN35: { layer.nextn.eh_proj = create_tensor(); } case LLM_ARCH_FOO:")),
        ("gap: graph", lambda: check_gap_graph("void build_mtp_head() { nextn(); }")),
        ("gap: spec-type", lambda: check_gap_spectype("COMMON_SPECULATIVE_TYPE_DRAFT_MTP,", "")),
        ("gap: spec-type flag", lambda: check_gap_spectype("", '"[none|draft-mtp]"')),
        ("model size", lambda: check_model_size(MODEL_BYTES - 1)),
        ("model hash", lambda: check_model_hash("0" * 64)),
        ("arch missing", lambda: check_gap_hparams("case LLM_ARCH_LLAMA:")),
        ("mtp upstream: bad repo", lambda: check_mtp_upstream("/nonexistent-repo-path")),
        ("subrepo: doc drift", lambda: check_subrepo_doc(
            "[subrepo]\n\tremote = https://example.com/r\n\tbranch = b\n\tcommit = deadbeef\n",
            "CLAUDE.md that never mentions the real values")),
        ("subrepo: empty .gitrepo", lambda: check_subrepo_doc("[subrepo]\n", "anything")),
        # The state gate must fail in BOTH directions and on partial rebases,
        # otherwise it is a one-way assertion that rots the moment work lands.
        ("state: declared absent, tree present", lambda: check_mtp_state(
            "case LLM_ARCH_QWEN35: { LLM_KV_NEXTN_PREDICT_LAYERS layer.nextn } case LLM_ARCH_X:",
            "nextn graph", "DRAFT_MTP", "draft-mtp", "absent")),
        ("state: declared present, tree absent", lambda: check_mtp_state(
            "case LLM_ARCH_QWEN35: { plain } case LLM_ARCH_X:", "", "", "", "present")),
        ("state: partial rebase", lambda: check_mtp_state(
            "case LLM_ARCH_QWEN35: { LLM_KV_NEXTN_PREDICT_LAYERS layer.nextn } case LLM_ARCH_X:",
            "", "", "", "present")),
        ("state: no marker declared", lambda: check_mtp_state(
            "case LLM_ARCH_QWEN35: { plain } case LLM_ARCH_X:", "", "", "", None)),
        # The head check must reject a requant that kept nextn.* but dropped the
        # MTP layer's own weights -- that file loads and then cannot draft.
        ("head: nextn without full block", lambda: check_mtp_head(
            "qwen35", 1, 65,
            ["blk.64.nextn.%s.weight" % s for s in NEXTN_SUFFIXES])),
        ("head: nextn tensors stripped", lambda: check_mtp_head(
            "qwen35", 1, 65,
            ["blk.64.%s.weight" % s for s in BLOCK_TENSORS])),
        ("head: wrong architecture", lambda: check_mtp_head("llama", 1, 65, [])),
        ("head: no nextn layers", lambda: check_mtp_head("qwen35", 0, 65, [])),
    ]
    bad = 0
    for name, fn in cases:
        ok, detail = fn()
        if ok:
            print("  NEGATIVE CONTROL FAILED  %-22s returned pass on broken input" % name)
            bad += 1
        else:
            print("  ok  %-22s correctly failed: %s" % (name, detail[:52]))
    print("\n%s: %d/%d checks reject broken input" % (
        "PASS" if not bad else "FAIL", len(cases) - bad, len(cases)))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", action="store_true", help="hash the model (slow, ~16.8 GB)")
    ap.add_argument("--base", metavar="UPSTREAM", help="path to an upstream llama.cpp clone")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    results, unchecked = [], []

    results.append(("subrepo doc matches .gitrepo", check_subrepo_doc(
        read(os.path.join(LC, ".gitrepo")), read(os.path.join(REPO, "CLAUDE.md")))))

    results.append(("turbo cache types", check_turbo_types(read(os.path.join(LC, "ggml/include/ggml.h")))))

    claude_md = read(os.path.join(REPO, "CLAUDE.md"))
    results.append(("MTP state vs declared stage", check_mtp_state(
        read(os.path.join(LC, "src/llama-model.cpp")),
        read(os.path.join(LC, "src/models/qwen35.cpp")),
        read(os.path.join(LC, "common/common.h")),
        read(os.path.join(LC, "common/arg.cpp")),
        declared_mtp_state(claude_md))))

    if os.path.exists(args.model):
        results.append(("model size", check_model_size(os.path.getsize(args.model))))
        try:
            results.append(("model MTP head", check_mtp_head(
                *read_gguf_mtp(args.model, os.path.join(LC, "gguf-py")))))
        except Exception as e:
            results.append(("model MTP head", (False, "could not read GGUF: %s" % e)))
        if args.hash:
            results.append(("model sha256", check_model_hash(sha256(args.model))))
        else:
            unchecked.append("model sha256 (pass --hash)")
    else:
        results.append(("model present", (False, "not found: %s" % args.model)))

    if args.base:
        results.append(("vendored base %s" % BASE_COMMIT, check_base(args.base, LC)))
        results.append(("upstream MTP fix", check_mtp_upstream(args.base)))
    else:
        unchecked.append("vendored base commit (pass --base UPSTREAM)")
        unchecked.append("upstream MTP fix %s (pass --base UPSTREAM)" % MTP_UPSTREAM)

    failed = 0
    for name, (ok, detail) in results:
        print("%-4s %-28s %s" % ("ok" if ok else "FAIL", name, detail))
        failed += not ok
    for u in unchecked:
        print("%-4s %s" % ("--", u))

    print("\n%d checked, %d failed, %d unchecked" % (len(results), failed, len(unchecked)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
