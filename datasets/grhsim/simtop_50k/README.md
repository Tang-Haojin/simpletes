# GrhSIM SimTop 50k auto-research bench

This bench evolves a schema-v2 structured JSON candidate whose payload is a
safe unified diff against the pinned `wolvrix` revision and whose
`candidate_mode` declares its attribution path. The evaluator never edits the
user checkout: it creates a locked local clone, verifies parent `be78e83` and
`wolvrix` `79ec203`, applies the patch only inside that slot, runs the
mode-specific generated-source attribution gates, builds a real local ELF,
runs fixed-ASLR function gates, and hands resolved artifacts to the trusted
runtime.

This pin pair is the fresh post-typed-state control namespace. Native
defaults already contain R (cold register-write guard admission/group hints), W (singleton
memory-write hints only inside an R-admitted run), and A (eligible adjacent
SystemTask/`xs_assert_v2` outer-guard nesting). They also contain cold
SystemTask/eligible standalone `xs_assert_v2` guards, always-inline hot word
helpers, direct in-range constant MemoryRead row loads without a redundant zero
pass, and cold dynamic scalar-shift/index bounds fallbacks. The MemoryFill F,
residual dynamic MemoryRead hint, and physical zero-tail mechanisms did not
land. The native principled hot-event mechanism structurally selects an input
event from reusable exact-posedge demand above a fixed cost, remaps it to typed
slot zero, predecodes exact posedge as a bool, and snapshots that bool per
covered batch while retaining enum handling for residual queries. Its gate does
not inspect port names, benchmark identity, ValueId, or a raw slot-count
threshold. Typed-state storage is also already native-default: persistent state
uses field-sensitive typed storage, persistent-state bool slots use native
`bool`, and materialized value-bucket bool slots use native `bool`. These are
three incremental mechanisms represented by four ablation nodes `B`, `S8`, `SB`,
and `SBV`; do not count the nodes as four optimizations or repropose them. None
of these landed or closed directions—including HS/TRBS under a new name—may be
reproposed.

The score is `control_mean_walltime_ms / candidate_mean_walltime_ms`. Absolute
control and candidate `Host time spent` values and their millisecond/percentage
delta are always returned. CPU, CCD, NUMA, PMU, generated C++, ELF, and profile
data are diagnostics, not substitute objectives. Retryable host-load or audit
failures are retried as the same candidate and do not count toward the valid
candidate budget.

## Start fresh, then continue only within the post-typed-state pins

The launcher fixes the requested model to Codex-compatible `k3` with reasoning
effort `ultra`, uses four RPUCG chains with `k=1`, runs four generation workers
in parallel while keeping evaluation serial, stops after eight valid generated
candidates or sixteen proposals, and disables reflection. Each K3 generation
has a three-hour (`10800 s`) timeout and may freely use up to three concurrently
open subagents, without prescribed roles. The subagent flag, cap, and prompt
apply only to K3; other models retain their native delegation behavior. It
sources the target checkout's `env.sh` before executing
SimpleTES and forces attribution verification, focused tests, and function
gates on, regardless of inherited shell settings. It passes only the paths of
the Kimi configuration files. The config is copied to the backend's private
temporary `CODEX_HOME`; the auth file is parsed without being copied, and its
API key is retained by a per-attempt loopback compatibility proxy for the active
`kimi` provider. Codex receives an unrelated ephemeral loopback credential;
generated tool shells inherit neither credential nor unrelated parent secrets.
Secrets are not placed in argv, logs, metrics, or checkpoints.

The first run after this repin must use the checked-in typed-state empty control seed shown
below. It creates a new pin-derived evaluator slot namespace and a fresh
checkpoint instance. Do not migrate, copy, resume, or seed from the old
`de37459`/`fd12d83`, `d31118b`/`16a9f49`, or `fbe4e1c`/`8f6ba14` namespaces;
those candidates are diffs against a different baseline and the launcher
rejects them fail-closed.

```bash
cd SimpleTES
source ../wolvrix-playground-gsim-calibrate-5/env.sh
./.venv/bin/python datasets/grhsim/simtop_50k/launcher.py --dry-run
./.venv/bin/python datasets/grhsim/simtop_50k/launcher.py --preflight-only
./.venv/bin/python datasets/grhsim/simtop_50k/launcher.py
```

`--preflight-only` performs one deterministic, repository-grounded Codex request
and exits. It forces one repository tool call, verifies a fresh nonce against an
immutable pinned blob, and requires an exact harmless include-order smoke diff.
The diff is checked with an isolated temporary Git index and object directory;
the checkout and its real Git objects, refs, and index are unchanged. This gate
has its own 600-second timeout, uses no repair attempts, does not construct the
SimpleTES engine or run the evaluator, and creates no checkpoint/instance. Use
it to validate K3 routing, credentials, tool use, JSON output, and patch
applicability without starting performance research. `--dry-run` remains
completely offline and only prints the future research command.

For an exact `Selected model is at capacity` failure, the Codex backend keeps
that request's private local session and sends `continue` to the exact
`thread.started.thread_id`. It never uses `--last`, so concurrent generation
workers cannot select one another's conversations. These bounded in-session
continuations have their own delay/budget and do not consume SimpleTES's normal
LLM retry count. Only after the continuation budget is exhausted (or no safe
thread ID was emitted) does the current `--codex-exec-retries` policy apply.
The default is three continuations with 1/2/5-second delays; set
`--codex-capacity-continuations 0` to disable them.

Remote-compaction failures and terminal reconnect/stream failures
(`Reconnecting...`, `stream disconnected before completion`, or
`Upstream request failed`) use the same exact-thread mechanism with a separate
budget. Their `continue` turns do not consume either the capacity budget or a
normal SimpleTES retry. After that budget is exhausted—or if the failed stream
did not emit a safe thread ID—the normal exec retry starts a fresh conversation.
The default is three transient continuations with 1/2/5-second delays; set
`--codex-transient-continuations 0` to disable them. Complete failure JSONL,
stderr, final-output, and classification metadata are retained for every failed
turn before any continuation is attempted.

Defaults:

- target checkout: sibling `wolvrix-playground-gsim-calibrate-5`
- model/provider: `k3` / `kimi`, reasoning effort `ultra`
- generation/evaluation concurrency: `4` / `1`
- generation timeout: `10800 s`
- maximum concurrently open K3 subagents per generation: `3`
- K3 context/model catalog: `k3_model_catalog.json` (`1000000` configured
  context tokens, `1048576` advertised maximum, `900000` auto-compact limit)
- config: `~/.codex/config.kimi.toml`
- auth: `~/.codex/auth.kimi.json`
- capability preflight timeout: `600 s` (independent of generation timeout)
- Codex capacity continuations / transient continuations / normal exec retries:
  `3` / `3` / `2`
- initial program: `datasets/grhsim/simtop_50k/init_program.txt`
- evaluator slots: `/tmp/simpletes-grhsim-simtop-50k`
- checkpoints: `SimpleTES/checkpoints/grhsim_simtop_50k/<timestamp>`

After a new-pin run exists, use `--resume CHECKPOINT`, and keep the same
target/slot roots, to continue a gracefully stopped schema-v2 run within its
original proposal budget. The launcher resolves and validates one exact
`db_state_*` plus its exact
`best_program.*`, then passes those paths to SimpleTES; it does not perform a
second "latest checkpoint" selection and no repeated `--init-program` is
needed. The launcher rejects legacy-schema seeds and resumes, and rejects a
checkpoint whose recorded evaluator metrics use different parent/wolvrix pins.
New checkpoints also record the non-sensitive Codex model effort, output mode,
tool-choice mode, config/repository path, and provider/local schema paths while
deliberately excluding the auth path and all API keys.
In particular, an old `de37459`/`fd12d83`, `d31118b`/`16a9f49`, or pre-RWA
checkpoint is rejected for both `--resume` and explicit `best_program.txt`
seeding.

```bash
./.venv/bin/python datasets/grhsim/simtop_50k/launcher.py \
  --resume checkpoints/grhsim_simtop_50k/<post-typed-state-run>/<date>/instance-<id>
```

An exhausted post-typed-state checkpoint may be extended in place only
with the explicit `--extend-resume-budget` opt-in. Both limits are absolute totals, not
increments. The engine preserves the existing nodes, scores, attempt/valid
counters, per-chain prompt counts, histories, and failure records; it
monotonically recomputes each chain's ceiling and writes the old/new limits plus
the extension point into subsequent checkpoint metadata. Shrinking either
limit, changing the chain count or `k`, selecting a limit below an already-used
counter, or passing the flag without `--resume` fails closed. For example, this
extends a completed `64/32` run to at most `192` total proposals while retaining
the same cumulative target of `32` valid candidates:

```bash
GRHSIM_INFRA_RETRIES=8 \
./.venv/bin/python datasets/grhsim/simtop_50k/launcher.py \
  --resume checkpoints/grhsim_simtop_50k/<post-typed-state-run>/<date>/instance-<id>/db_state_<time> \
  --extend-resume-budget \
  --max-proposals 192 \
  --valid-target 32
```

Use a fresh instance seeded by the same-pin `best_program.txt` instead when an
independent continuation is desired rather than preserving the original search
tree and counters:

```bash
./.venv/bin/python datasets/grhsim/simtop_50k/launcher.py \
  --init-program checkpoints/grhsim_simtop_50k/<post-typed-state-run>/<date>/instance-<id>/db_state_<time>/best_program.txt
```

The seed must be a regular, non-symlink file containing a complete marked
schema-v2 candidate document. A checkpoint `best_program.txt` seed must also
match a sibling node whose recorded parent/wolvrix pins equal this evaluator's
new `be78e83`/`79ec203` pins. It is evaluated again as the new instance's
initial node; that initial evaluation does not consume a proposal or
valid-candidate slot.
Omit `--resume` for this continuation so the launcher creates a fresh checkpoint
instance with a fresh bounded search budget. Build/perf/staging/checkpoint
artifacts are deliberately untracked.

The default budget remains `16` proposals / `8` valid candidates. Fresh runs
may explicitly request up to `64` proposals. An explicit in-place extension may
raise the absolute proposal ceiling to `256`; keep the valid target at or below
the proposal budget. For example, a doubled fresh search uses:

```bash
GRHSIM_INFRA_RETRIES=4 \
./.venv/bin/python datasets/grhsim/simtop_50k/launcher.py \
  --init-program checkpoints/grhsim_simtop_50k/<post-typed-state-run>/<date>/instance-<id>/db_state_<time>/best_program.txt \
  --max-proposals 32 \
  --valid-target 16 \
  --llm-timeout 10800 \
  --gen-concurrency 4
```

`GRHSIM_INFRA_RETRIES=4` keeps up to five quiet-CCD runtime attempts inside one
already-built evaluator invocation. It does not relax any runtime gate. A fresh
continuation omits `--resume`; an in-place continuation requires both an exact
resume state and the explicit extension flag above.

If all in-process runtime attempts fail only because no whole CCD stays quiet,
reuse the already-gated artifacts with the runtime-only entry point:

```bash
GRHSIM_INFRA_RETRIES=4 GRHSIM_BUILD_JOBS=4 \
./.venv/bin/python datasets/grhsim/simtop_50k/retry_runtime.py \
  /path/to/the/exact/candidate.txt \
  --source-repo /path/to/wolvrix-playground-gsim-calibrate-5 \
  --slot-root /tmp/simpletes-grhsim-simtop-50k
```

The full evaluator writes a versioned proof marker only after mode-specific
attribution, focused, and 100/10k function gates pass. Before the candidate
build it snapshots the control artifact and `env.sh` SHA-256 identities in
memory; before proof publication or runtime it revalidates that control and
requires candidate image/NEMU identity to match. Each runtime result is
committed as an immutable attempt whose JSON files and candidate proof are
bound by full SHA-256 values, including the candidate mode and the exact
control binary/image/NEMU plus generated/build/toolchain identity. The attempt
binds that control incarnation transitively through the full proof SHA-256.
`retry_runtime.py` holds the same slot lock and revalidates the candidate
mode/patch/options, pinned revisions, generated output,
build/toolchain identity, `env.sh`, binary, image, NEMU, and the latest complete
retryable attempt before invoking the normal ABBA/BAAB path. It never clones,
emits, builds, or falls back to rebuilding an invalid cache. Results created by
older evaluator revisions without this proof are deliberately not reusable.

## Candidate and promotion rules

`candidate.schema.json` is the schema-v2 model output contract and permits only
the two non-control proposal modes. The evaluator parser separately accepts the
`control` shape as an init-only special case, so the pinned seed and
`--validate-only` remain usable without allowing an LLM to propose another
control. The provider-facing schema is deliberately a flat object without
cross-field `oneOf` conditionals or size-bound keywords unsupported by the
Codex structured-output API. Field descriptions state the mode rules, while
the evaluator remains the authoritative fail-closed enforcement point for
non-empty patch/options shape, cardinality, size, and value safety. Patch paths
are limited to existing tracked C/C++ source/header files under `wolvrix/lib`,
`wolvrix/include`, and `wolvrix/app/pybind`; additions, deletions, binary,
symlink, executable, rename, traversal, generated-output, submodule, harness,
build-description, test, and secret-bearing patches are rejected before
`git apply`. Model output represents `enable_options` as name/value entries;
the evaluator normalizes them into an explicit pinned optimization allowlist.
resume, stats, profile, stop, export, instrumentation, measurement, and
diagnostic policy values are unavailable.

The candidate mode and patch/options shape are cross-checked. Only the latter
two modes are legal model output:

- `control` has no patch and no options and is reserved for the initial seed.
- `default-path` has a patch and no options. The patched native `options={}`
  generated fingerprint must differ from the unpatched current-default control.
- `explicit-options` has both a patch and options. Its patched no-options output
  must equal control, while its enabled output must differ from both control and
  an unpatched same-options emit. Thus an old knob alone cannot be credited to a
  no-op patch.

`active_mask_gap_pack_policy=targeted-direct` controls only non-table direct
active-mask gap packing. It does not control exact event/commit behavior and
must not be borrowed as a generic optimization gate. That independent mechanism
uses `commit_exact_event_policy`; refinements to behavior already selected by
the native C++ default belong in `default-path` mode.

The landed R/W/A, four-positive, and principled hot-event source is part of that
native control, not candidate space. The unlanded F MemoryFill, residual
MemoryRead, and physical zero-tail mechanisms are closed as well. Do not submit
any candidate that recreates those mechanisms under another patch shape or
option. The
`materialize_ablation.py --compose-rwa` path remains pinned to historical
Wolvrix `8f6ba14` solely to reproduce prior ablation evidence; it does not
define the current evaluator baseline and its output must not seed this
namespace.

The runtime performs a quiet, dynamically selected CCD-local ABBA 50k screen
with ASLR disabled. A positive screen is repeated in reversed BAAB order before
its pooled score is used. Only candidates with consistent positive walltime are
eligible for later code retention/default promotion. Promotion documentation
and commits belong in `pdocs/grhsim_opt_thj` under its `RULES.md`; evaluator
artifacts themselves must not be committed.

Before any candidate emit, the evaluator copies the cached control's generated
`.sv`/`.v` RTL and XiangShan difftest generated sources into private candidate
files. A content manifest is checked after every phase used by the selected
mode. This keeps attribution comparisons on identical elaboration inputs while
still preventing a candidate from mutating the cached control.

The cached control is reused only when its pinned revisions, copied `env.sh`,
fixed build configuration, resolved toolchain fingerprint, generated-source
fingerprint, and SHA-256 identities of the ELF, image, and NEMU all match.
