# FSDP PEFT checkpoint validation

This records the user-run Kaggle results reported on 2026-11-11 for the source fix in
`fc992a06b606b7d9411c65ac60102bcc76f17479` and the reproducer introduced in `93915e5c7`.
The GPU run was performed by the user, not on the local macOS development machine.

## Environment

- Hardware: two NVIDIA T4 GPUs on Kaggle.
- Previously supplied package versions: PyTorch `2.11.0+cu130`, Accelerate `1.15.0`, PEFT `0.21.2`.
- Model: random two-layer Llama with LoRA on `q_proj` and `v_proj`, rank 8, alpha 16.
- Reproducer defaults: float32, two processes, FSDP1 and `SHARDED_STATE_DICT`; individual cases override these options.
- Distributed debug logging now defaults to `INFO`; an explicit environment setting is preserved.
- Pytest runs serially; the test harness launches its own training workers.

```bash
RUN_SLOW=1 python -m pytest -p xdist -n 0 \
  tests/trainer/distributed/test_fsdp_peft_resume.py -v
```

## Reported matrix

### Full 10-case test matrix (Initial Run with default `TORCH_DISTRIBUTED_DEBUG=DETAIL`)

```text
2 failed, 7 passed, 1 skipped, 2 warnings in 256.42s
```

All names below belong to `TestFSDPPeftResume`.

| Test | Result |
| --- | --- |
| `test_full_parameter_fsdp_control` | PASSED |
| `test_load_best_model_0_full_fsdp1` | PASSED |
| `test_load_best_model_1_sharded_fsdp1` | PASSED |
| `test_load_best_model_2_sharded_fsdp2` | FAILED (worker traceback not supplied) |
| `test_named_adapters` | PASSED |
| `test_resume_0_full_fsdp1` | PASSED |
| `test_resume_1_sharded_fsdp1` | PASSED |
| `test_resume_2_sharded_fsdp2` | FAILED (`RuntimeError: Backend nccl does not support allgather_into_tensor_coalesced`) |
| `test_single_gpu_peft_control` | PASSED |
| `test_zero3_peft_control` | SKIPPED |

### FSDP2 Verified Diagnostic Rerun (`TORCH_DISTRIBUTED_DEBUG=INFO`)

Executed on two NVIDIA T4 GPUs using the exact committed 355-line reproducer harness with `TORCH_DISTRIBUTED_DEBUG=INFO`:

```bash
TORCH_DISTRIBUTED_DEBUG=INFO \
  FSDP_PEFT_LOG_DIR=/kaggle/working/fsdp-peft-info-logs RUN_SLOW=1 \
  python -m pytest -p xdist -n 0 -s \
  tests/trainer/distributed/test_fsdp_peft_resume.py -k sharded_fsdp2 -vv
```

```text
============ 2 passed, 8 deselected, 2 warnings in 83.28s (0:01:23) ============
Pytest exit code: 0
Worker exceptions: (none)
```

| Test | Result | Notes |
| --- | --- | --- |
| `test_load_best_model_2_sharded_fsdp2` | PASSED (43.56s) | `PASS: load_best_model_at_end restored checkpoint-2` |
| `test_resume_2_sharded_fsdp2` | PASSED (39.65s) | `PASS: resumed weights match uninterrupted four-step training` |

### Consolidated Matrix Status

- FSDP1: six tested cases passed, including the full-parameter control.
- FSDP2: both tested cases passed on the `INFO` rerun.
- Single-GPU PEFT control: passed.
- ZeRO-3 DeepSpeed control: skipped because the backend was not installed.
- Across the two runs, nine distinct cases have passed and one remains skipped.
  This is a consolidation of the latest outcomes, not a single `9 passed, 1 skipped`
  pytest run. The initial failures remain recorded above.

The seven passing tests comprise all six tested FSDP1 cases and the single-GPU PEFT control.
They cover full and sharded checkpoint resume, best-model restoration, the named-adapter
case, and full-parameter FSDP resume on this model and environment.

## Assertions exercised

The resume scenarios use fresh training processes. Before resuming updates, the harness
compares loaded weights with the step-2 snapshot, checks AdamW step counters equal 2, and
checks the restored scheduler epoch equals 2. It then requires step 4 to complete and
compares final weights with uninterrupted four-step training using `rtol=1e-5`, `atol=1e-7`.
PEFT comparisons cover the selected adapter; the full-parameter control compares the
model state. Best-model scenarios compare the final weights with the reference for the
selected checkpoint. No loaders, checkpoints, or distributed barriers are mocked.

The original two-GPU failure had empty optimizer state on both ranks and scheduler epoch 2.
After the fix, the sharded FSDP1 resume scenario passed these assertions. This supports
the metadata-allocation fix for the observed failure; it does not establish correctness
for every possible FSDP1 configuration or a performance improvement.

## Failures and verification limits

The subsequent supplied pytest output shows the resume test's controller failed in
phase `save`, while the best-model test's controller failed in phase `best`. This
corrects the earlier summary that both failed during the initial save phase. Phase
labels identify whole training jobs, not the precise failing operation within a worker.

The user subsequently supplied the first worker traceback from `save.log`. After
training completed, the harness called `trainer.accelerator.get_state_dict()` to
collect validation weights. FSDP2 DTensor redistribution then failed with:

```text
RuntimeError: Backend nccl does not support allgather_into_tensor_coalesced
```

This identifies a failure in the harness's post-training snapshot; it does not show
that Trainer checkpoint saving or optimizer restoration failed. The supplied traceback
is from the resume scenario's `save` job. The best-model job's worker traceback has
not been supplied, so the two failures cannot yet be assigned the same cause.

The earlier harness default was `TORCH_DISTRIBUTED_DEBUG=DETAIL`. PyTorch
[issue #137505](https://github.com/pytorch/pytorch/issues/137505) reports this exact
error with that setting and successful runs after removing it. `DETAIL` enables a
process-group wrapper whose collective support can differ from the underlying NCCL
backend. Both FSDP2 cases subsequently passed with `INFO` and the canonical harness,
supporting the debug-wrapper explanation for the supplied snapshot error. The harness
also changed between the shortened failing script and the successful rerun, so these
runs do not isolate the environment setting as the sole changed variable.

A matching unmodified-baseline run has not been supplied. The existing FSDP/PEFT TODO
comment does not establish the cause or make either failure expected. Both initial
failures remain visible in the historical matrix; neither was skipped or marked as expected.

The reproducer records worker exceptions through TorchElastic and retains
per-rank stdout/stderr. Rerunning the two FSDP2 cases with `TORCH_DISTRIBUTED_DEBUG=INFO`
completed cleanly:

- `test_load_best_model_2_sharded_fsdp2`: PASSED in 43.56s (`PASS: load_best_model_at_end restored checkpoint-2`)
- `test_resume_2_sharded_fsdp2`: PASSED in 39.65s (`PASS: resumed weights match uninterrupted four-step training`)
- Zero worker exceptions recorded (`Worker exceptions: None`).

The supplied output confirms `DebugLevel.INFO`, PyTorch `2.11.0+cu130`, two CUDA GPUs,
and pytest exit code zero. Final resumed adapter weights matched uninterrupted training
within `rtol=1e-5`, `atol=1e-7`; this is a tolerance-based comparison, not bit-exact
convergence. Best-model loading selected and restored checkpoint 2. No production
FSDP2 loader changes, skipped cases, or relaxed assertions were needed.

Retained logs: `/kaggle/working/fsdp-peft-info-logs-20261111T115509713623Z/`.
The harness now defaults to the validated `INFO` setting and records the effective
debug level at training start. Users can still explicitly opt into `DETAIL`.

The DeepSpeed control was skipped because DeepSpeed was not installed. ZeRO-3 behavior
was therefore not GPU-verified by this run.

Across all tested distributed configurations on two NVIDIA T4 GPUs:
- FSDP1: 6 / 6 PASSED
- FSDP2: 2 / 2 PASSED
- Single-GPU Control: 1 / 1 PASSED
- Latest outcomes across the initial run and FSDP2 rerun: nine distinct cases passed,
  one skipped. A complete matrix run with the new default has not been supplied.

No PR, push, maintainer acceptance, or upstream attribution is implied by this record.
The branch is held locally awaiting maintainer review on huggingface/transformers#50549.
