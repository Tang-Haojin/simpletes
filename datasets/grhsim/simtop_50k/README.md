# GrhSIM SimTop 50k auto-research bench

This bench evolves a structured JSON candidate whose payload is a safe unified
diff against the pinned `wolvrix` revision. The evaluator never edits the user
checkout: it creates a locked local clone, verifies parent `b90d204` and
`wolvrix` `f17e90e`, applies the patch only inside that slot, checks default-off
generated-source identity, builds a real local ELF, runs fixed-ASLR function
gates, and hands resolved artifacts to the trusted runtime.

The score is `control_mean_walltime_ms / candidate_mean_walltime_ms`. Absolute
control and candidate `Host time spent` values and their millisecond/percentage
delta are always returned. CPU, CCD, NUMA, PMU, generated C++, ELF, and profile
data are diagnostics, not substitute objectives. Retryable host-load or audit
failures are retried as the same candidate and do not count toward the valid
candidate budget.

## Start, resume, or continue from the best candidate

The launcher fixes the requested model to `gpt-5.6-sol` with reasoning effort
`ultra`, uses four RPUCG chains with `k=1`, serial generation/evaluation, stops
after eight valid generated candidates or sixteen proposals, and disables
reflection. It sources the target checkout's `env.sh` before executing
SimpleTES and forces default-off verification, focused tests, and function
gates on, regardless of inherited shell settings. It passes only the paths of
the MJY configuration files; their
contents are copied to the backend's private temporary `CODEX_HOME` and are not
placed in argv, environment variables, logs, metrics, or checkpoints.

```bash
cd SimpleTES
source ../wolvrix-playground-gsim-calibrate-5/env.sh
./.venv/bin/python datasets/grhsim/simtop_50k/launcher.py --dry-run
./.venv/bin/python datasets/grhsim/simtop_50k/launcher.py
```

Defaults:

- target checkout: sibling `wolvrix-playground-gsim-calibrate-5`
- config: `~/.codex/config.mjy.toml`
- auth: `~/.codex/auth.mjy.json`
- initial program: `datasets/grhsim/simtop_50k/init_program.txt`
- evaluator slots: `/tmp/simpletes-grhsim-simtop-50k`
- checkpoints: `SimpleTES/checkpoints/grhsim_simtop_50k/<timestamp>`

Use `--resume CHECKPOINT`, and keep the same target/slot roots, to continue a
gracefully stopped run within its original proposal budget. When resuming a run
that was itself best-seeded, repeat the same `--init-program` argument because
runtime configuration is supplied by the current launcher invocation.

```bash
./.venv/bin/python datasets/grhsim/simtop_50k/launcher.py \
  --resume checkpoints/grhsim_simtop_50k/<run>/<date>/instance-<id>
```

Once a run has exhausted its proposal budget, start a new instance seeded by
the previous instance's `best_program.txt` instead of trying to enlarge the old
checkpoint's chain budgets:

```bash
./.venv/bin/python datasets/grhsim/simtop_50k/launcher.py \
  --init-program checkpoints/grhsim_simtop_50k/<run>/<date>/instance-<id>/db_state_<time>/best_program.txt
```

The seed must be a regular, non-symlink file containing a complete marked
candidate document. It is evaluated again as the new instance's initial node;
that initial evaluation does not consume a proposal or valid-candidate slot.
Omit `--resume` for this continuation so the launcher creates a fresh checkpoint
instance with a fresh bounded search budget. Build/perf/staging/checkpoint
artifacts are deliberately untracked.

The default budget remains `16` proposals / `8` valid candidates. Longer fresh
continuations may explicitly request up to `64` proposals; keep the valid target
at or below the proposal budget. For example, a doubled search uses:

```bash
GRHSIM_INFRA_RETRIES=4 \
./.venv/bin/python datasets/grhsim/simtop_50k/launcher.py \
  --init-program checkpoints/grhsim_simtop_50k/<run>/<date>/instance-<id>/db_state_<time>/best_program.txt \
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

The full evaluator writes a versioned proof marker only after attribution,
default-off, focused, and 100/10k function gates pass. Each runtime result is
committed as an immutable attempt whose JSON files and candidate proof are
bound by full SHA-256 values. `retry_runtime.py` holds the same slot lock and
revalidates the candidate patch/options, pinned revisions, generated output,
build/toolchain identity, `env.sh`, binary, image, NEMU, and the latest complete
retryable attempt before invoking the normal ABBA/BAAB path. It never clones,
emits, builds, or falls back to rebuilding an invalid cache. Results created by
older evaluator revisions without this proof are deliberately not reusable.

## Candidate and promotion rules

`candidate.schema.json` is the model output contract. Patch paths are limited
to existing tracked C/C++ source/header files under `wolvrix/lib`,
`wolvrix/include`, and `wolvrix/app/pybind`; additions, deletions, binary,
symlink, executable, rename, traversal, generated-output, submodule, harness,
build-description, test, and secret-bearing patches are rejected before
`git apply`. Model output represents `enable_options` as name/value entries;
the evaluator normalizes them into an explicit pinned optimization allowlist.
resume, stats, profile, stop, export, instrumentation, measurement, and
diagnostic policy values are unavailable.

Every generated candidate must be explicitly enabled and byte-identical to the
current-default generated C++/headers while disabled. An additional unpatched
build with the same options must differ from the enabled patched output, so an
old knob alone cannot be credited to a no-op patch. The runtime performs a
quiet, dynamically selected CCD-local ABBA 50k screen with ASLR disabled. A
positive screen is repeated in reversed BAAB order before its pooled score is
used. Only candidates with consistent positive walltime are eligible for later
code retention/default promotion. Promotion documentation and commits belong
in `pdocs/grhsim_opt_thj` under its `RULES.md`; evaluator artifacts themselves
must not be committed.

Before any candidate emit, the evaluator copies the cached control's generated
`.sv`/`.v` RTL and XiangShan difftest generated sources into private candidate
files. A content manifest is checked after the unpatched-options, disabled, and
enabled phases. This keeps the byte-exact default-off comparison on identical
elaboration inputs while still preventing a candidate from mutating the cached
control.

The cached control is reused only when its pinned revisions, copied `env.sh`,
fixed build configuration, resolved toolchain fingerprint, generated-source
fingerprint, and SHA-256 identities of the ELF, image, and NEMU all match.
