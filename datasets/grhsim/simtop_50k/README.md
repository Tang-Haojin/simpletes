# GrhSIM SimTop 50k auto-research bench

This bench evolves a schema-v2 structured JSON candidate whose payload is a
safe unified diff against the pinned `wolvrix` revision and whose
`candidate_mode` declares its attribution path. The evaluator never edits the
user checkout: it creates a locked local clone, verifies parent `d31118b` and
`wolvrix` `16a9f49`, applies the patch only inside that slot, runs the
mode-specific generated-source attribution gates, builds a real local ELF,
runs fixed-ASLR function gates, and hands resolved artifacts to the trusted
runtime.

This pin pair is the fresh post-RWA control namespace. Native defaults already
contain R (cold register-write guard admission/group hints), W (singleton
memory-write hints only inside an R-admitted run), and A (eligible adjacent
SystemTask/`xs_assert_v2` outer-guard nesting). The MemoryFill F tier did not
land. R/W/A must not be rediscovered or claimed by a candidate, and F is a
closed prior direction that must not be reproposed.

The score is `control_mean_walltime_ms / candidate_mean_walltime_ms`. Absolute
control and candidate `Host time spent` values and their millisecond/percentage
delta are always returned. CPU, CCD, NUMA, PMU, generated C++, ELF, and profile
data are diagnostics, not substitute objectives. Retryable host-load or audit
failures are retried as the same candidate and do not count toward the valid
candidate budget.

## Start fresh, then continue only within the post-RWA pins

The launcher fixes the requested model to Codex-compatible `k3` with reasoning
effort `ultra`, uses four RPUCG chains with `k=1`, serial generation/evaluation, stops
after eight valid generated candidates or sixteen proposals, and disables
reflection. It sources the target checkout's `env.sh` before executing
SimpleTES and forces attribution verification, focused tests, and function
gates on, regardless of inherited shell settings. It passes only the paths of
the Kimi configuration files. The config is copied to the backend's private
temporary `CODEX_HOME`; the auth file is parsed without being copied, and its
API key is retained by a per-attempt loopback compatibility proxy for the active
`kimi` provider. Codex receives an unrelated ephemeral loopback credential;
generated tool shells inherit neither credential nor unrelated parent secrets.
Secrets are not placed in argv, logs, metrics, or checkpoints.

The first run after this repin must use the checked-in empty control seed shown
below. It creates a new pin-derived evaluator slot namespace and a fresh
checkpoint instance. Do not migrate, copy, resume, or seed from the old
`fbe4e1c`/`8f6ba14` namespace; those candidates are diffs against a different
baseline and the launcher rejects them fail-closed.

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

Defaults:

- target checkout: sibling `wolvrix-playground-gsim-calibrate-5`
- model/provider: `k3` / `kimi`, reasoning effort `ultra`
- config: `~/.codex/config.kimi.toml`
- auth: `~/.codex/auth.kimi.json`
- capability preflight timeout: `600 s` (independent of generation timeout)
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
In particular, an old pre-RWA checkpoint is rejected for both `--resume` and
explicit `best_program.txt` seeding.

```bash
./.venv/bin/python datasets/grhsim/simtop_50k/launcher.py \
  --resume checkpoints/grhsim_simtop_50k/<post-rwa-run>/<date>/instance-<id>
```

Once a post-RWA run has exhausted its proposal budget, start a new instance
seeded by that same-pin instance's `best_program.txt` instead of trying to
enlarge the checkpoint's chain budgets:

```bash
./.venv/bin/python datasets/grhsim/simtop_50k/launcher.py \
  --init-program checkpoints/grhsim_simtop_50k/<post-rwa-run>/<date>/instance-<id>/db_state_<time>/best_program.txt
```

The seed must be a regular, non-symlink file containing a complete marked
schema-v2 candidate document. A checkpoint `best_program.txt` seed must also
match a sibling node whose recorded parent/wolvrix pins equal this evaluator's
new `d31118b`/`16a9f49` pins. It is evaluated again as the new instance's
initial node; that initial evaluation does not consume a proposal or
valid-candidate slot.
Omit `--resume` for this continuation so the launcher creates a fresh checkpoint
instance with a fresh bounded search budget. Build/perf/staging/checkpoint
artifacts are deliberately untracked.

The default budget remains `16` proposals / `8` valid candidates. Longer fresh
continuations may explicitly request up to `64` proposals; keep the valid target
at or below the proposal budget. For example, a doubled search uses:

```bash
GRHSIM_INFRA_RETRIES=4 \
./.venv/bin/python datasets/grhsim/simtop_50k/launcher.py \
  --init-program checkpoints/grhsim_simtop_50k/<post-rwa-run>/<date>/instance-<id>/db_state_<time>/best_program.txt \
  --max-proposals 32 \
  --valid-target 16 \
  --llm-timeout 5400
```

`GRHSIM_INFRA_RETRIES=4` keeps up to five quiet-CCD runtime attempts inside one
already-built evaluator invocation. It does not relax any runtime gate. A fresh
long continuation must still omit `--resume`, so the larger limits apply to a
new instance rather than mutating an exhausted checkpoint.

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

The landed R/W/A source is part of that native control, not candidate space.
The unlanded F MemoryFill tier is closed as well. Do not submit any candidate
that recreates R, W, A, or F under another patch shape or option. The
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
