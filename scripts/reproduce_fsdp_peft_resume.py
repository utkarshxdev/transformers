# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Real-process checkpoint investigation for Transformers #50549 / PEFT #4132.

Run the complete two-GPU experiment, with fresh processes for each training run:

    python scripts/reproduce_fsdp_peft_resume.py --output-dir ./tmp_fsdp_peft_output

Or run the interrupted and resumed jobs individually:

    torchrun --standalone --nproc_per_node=2 scripts/reproduce_fsdp_peft_resume.py --phase save
    torchrun --standalone --nproc_per_node=2 scripts/reproduce_fsdp_peft_resume.py --phase resume

Use --fsdp-version 2 for FSDP2, --state-dict-type FULL_STATE_DICT for FSDP1 full
checkpoints, --no-peft for full-parameter FSDP, or --backend deepspeed for ZeRO-3.
The explicitly selected --backend plain --cpu --processes 1 control works on a
CPU; it does not reproduce or verify FSDP/NCCL behavior.

No checkpoint files are fabricated and no distributed functions are patched.
Failures and watchdog timeouts are preserved in the per-phase log files.
Set FSDP_PEFT_LOG_DIR to a persistent directory to retain phase logs and per-rank
torchrun output even when pytest removes its temporary experiment directories.
"""

import argparse
import copy
import faulthandler
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("all", "save", "resume", "reference", "best"), default="all")
    parser.add_argument("--scenario", choices=("resume", "best", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("./tmp_fsdp_peft_output"))
    parser.add_argument("--backend", choices=("fsdp", "plain", "deepspeed"), default="fsdp")
    parser.add_argument("--fsdp-version", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--state-dict-type", choices=("FULL_STATE_DICT", "SHARDED_STATE_DICT"), default="SHARDED_STATE_DICT"
    )
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument(
        "--timeout", type=int, default=240, help="Per-job watchdog in seconds; collective timeout is 60s."
    )
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--adapter-name", default="default")
    parser.add_argument("--extra-adapter", action="store_true")
    parser.add_argument("--no-peft", action="store_true")
    parser.add_argument("--resume-from-checkpoint", choices=("auto", "path"), default="auto")
    parser.add_argument("--cpu", action="store_true", help="Explicit single-process plain-Trainer control only.")
    args = parser.parse_args()
    if args.processes < 1 or args.timeout < 1:
        parser.error("--processes and --timeout must be positive")
    if args.cpu and (args.backend != "plain" or args.processes != 1):
        parser.error("--cpu requires --backend plain --processes 1; it does not emulate FSDP")
    if args.backend == "fsdp" and args.fsdp_version == 2 and args.state_dict_type == "FULL_STATE_DICT":
        parser.error("Use SHARDED_STATE_DICT for the FSDP2 test matrix")
    return args


def launch(args, phase, output_dir):
    """Run a real job and kill its entire process group on timeout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir
    if os.environ.get("FSDP_PEFT_LOG_DIR"):
        log_dir = Path(os.environ["FSDP_PEFT_LOG_DIR"]) / args.output_dir.name / output_dir.name
    log_dir.mkdir(parents=True, exist_ok=True)
    rank_log_dir = log_dir / f"{phase}-workers"
    cmd = [sys.executable]
    if not args.cpu:
        cmd.extend(
            [
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--local-addr=127.0.0.1",
                f"--nproc_per_node={args.processes}",
                f"--log-dir={rank_log_dir}",
                "--tee=3",
            ]
        )
    cmd.extend(
        [
            str(Path(__file__).resolve()),
            "--phase",
            phase,
            "--output-dir",
            str(output_dir),
            "--backend",
            args.backend,
            "--fsdp-version",
            str(args.fsdp_version),
            "--state-dict-type",
            args.state_dict_type,
            "--processes",
            str(args.processes),
            "--dtype",
            args.dtype,
            "--adapter-name",
            args.adapter_name,
            "--resume-from-checkpoint",
            args.resume_from_checkpoint,
        ]
    )
    for flag in ("cpu", "no_peft", "extra_adapter"):
        if getattr(args, flag):
            cmd.append("--" + flag.replace("_", "-"))
    log_file = log_dir / f"{phase}.log"
    print(
        json.dumps({"phase": phase, "command": cmd, "log": str(log_file), "rank_logs": str(rank_log_dir)}), flush=True
    )
    with log_file.open("w") as log:
        with subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True) as process:
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise SystemExit(f"Watchdog expired after {args.timeout}s in {phase}; inspect {log_file}") from None
    if code:
        print(log_file.read_text(), file=sys.stderr)
        for error_file in sorted(rank_log_dir.rglob("error.json")):
            print(f"Recorded worker exception: {error_file}", file=sys.stderr)
        raise SystemExit(f"{phase} exited with code {code}; inspect {log_file}")


def run_experiment(args):
    import torch
    from safetensors.torch import load_file

    if "RANK" in os.environ:
        raise SystemExit("Use --phase save/resume/best/reference under torchrun; run --phase all with plain Python.")
    interrupted = args.output_dir / "interrupted"
    reference = args.output_dir / "reference"
    best = args.output_dir / "best"
    if any(path.exists() for path in (interrupted, reference, best)):
        raise SystemExit("Choose a fresh --output-dir; existing experiment directories will not be overwritten.")
    launch(args, "save", interrupted)
    launch(args, "reference", reference)
    if args.scenario in ("resume", "all"):
        launch(args, "resume", interrupted)
        torch.testing.assert_close(
            load_file(str(interrupted / "resume-weights.safetensors")),
            load_file(str(reference / "reference-weights.safetensors")),
            rtol=1e-5,
            atol=1e-7,
        )
        print("PASS: resumed weights match uninterrupted four-step training", flush=True)
    if args.scenario in ("best", "all"):
        launch(args, "best", best)
        report = json.loads((best / "best-result.json").read_text())
        best_step = int(Path(report["best_model_checkpoint"]).name.split("-")[-1])
        expected = (
            interrupted / "save-weights.safetensors" if best_step == 2 else reference / "reference-weights.safetensors"
        )
        torch.testing.assert_close(
            load_file(str(best / "best-weights.safetensors")), load_file(str(expected)), rtol=1e-5, atol=1e-7
        )
        print(f"PASS: load_best_model_at_end restored checkpoint-{best_step}", flush=True)


def run_worker(args):
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from safetensors.torch import load_file, save_file

    from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainerCallback, TrainingArguments, set_seed

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != args.processes:
        raise SystemExit(
            f"Expected {args.processes} ranks, got WORLD_SIZE={world_size}; launch this phase with torchrun."
        )
    if args.phase != "resume" and any(args.output_dir.glob("checkpoint-*")):
        raise SystemExit("Choose a fresh --output-dir; existing checkpoints will not be overwritten.")
    set_seed(42)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_dropout=0.0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            attn_implementation="eager",
        )
    ).to(getattr(torch, args.dtype))
    if not args.no_peft:
        config = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config, adapter_name=args.adapter_name, autocast_adapter_dtype=False)
        if args.extra_adapter:
            model.add_adapter("inactive", copy.deepcopy(config))
            model.set_adapter(args.adapter_name)
    generator = torch.Generator().manual_seed(123)
    tokens = torch.randint(3, 64, (32, 16), generator=generator)
    dataset = [{"input_ids": row, "attention_mask": torch.ones_like(row), "labels": row.clone()} for row in tokens]
    backend_kwargs = {}
    if args.backend == "fsdp":
        backend_kwargs = {
            "fsdp": "full_shard auto_wrap",
            "fsdp_config": {
                "version": args.fsdp_version,
                "state_dict_type": args.state_dict_type,
                "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            },
        }
    elif args.backend == "deepspeed":
        backend_kwargs = {
            "deepspeed": {
                "train_micro_batch_size_per_gpu": "auto",
                "train_batch_size": "auto",
                "gradient_accumulation_steps": "auto",
                "zero_optimization": {"stage": 3, "stage3_gather_16bit_weights_on_model_save": True},
            }
        }
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-3,
        optim="adamw_torch",
        save_strategy="steps",
        save_steps=2,
        logging_steps=1,
        report_to="none",
        disable_tqdm=True,
        eval_strategy="steps" if args.phase == "best" else "no",
        eval_steps=2,
        load_best_model_at_end=args.phase == "best",
        metric_for_best_model="eval_loss",
        greater_is_better=True,
        seed=42,
        data_seed=42,
        use_cpu=args.cpu,
        ddp_timeout=60,
        **backend_kwargs,
    )
    trainer = None

    def snapshot():
        # Every rank participates in real state-dict collectives; only rank zero writes.
        state = trainer.accelerator.get_state_dict(trainer.model_wrapped)
        if not trainer.is_world_process_zero():
            return None
        if not args.no_peft:
            state = get_peft_model_state_dict(
                trainer.accelerator.unwrap_model(trainer.model), state_dict=state, adapter_name=args.adapter_name
            )
        return {name: tensor.detach().cpu().contiguous().clone() for name, tensor in state.items()}

    class ObserveCheckpoint(TrainerCallback):
        def on_train_begin(self, training_args, state, control, optimizer=None, lr_scheduler=None, **kwargs):
            if args.phase == "resume":
                loaded = snapshot()
                if loaded is not None:
                    torch.testing.assert_close(
                        loaded, load_file(str(args.output_dir / "save-weights.safetensors")), rtol=0, atol=0
                    )
                scheduler = getattr(lr_scheduler, "scheduler", lr_scheduler)
                if args.backend == "deepspeed":
                    steps = [trainer.model_wrapped.global_steps]
                else:
                    steps = [int(value["step"].item()) for value in optimizer.state.values() if "step" in value]
                if not steps or any(step != 2 for step in steps) or scheduler.last_epoch != 2:
                    raise AssertionError(
                        f"Optimizer/scheduler were not restored: steps={steps}, epoch={scheduler.last_epoch}"
                    )

        def on_step_end(self, training_args, state, control, **kwargs):
            if args.phase == "save" and state.global_step == 2:
                control.should_training_stop = True

    trainer = Trainer(
        model=model, args=training_args, train_dataset=dataset, eval_dataset=dataset, callbacks=[ObserveCheckpoint()]
    )
    resume = None
    if args.phase == "resume":
        resume = True if args.resume_from_checkpoint == "auto" else str(args.output_dir / "checkpoint-2")
    print(
        json.dumps(
            {"phase": args.phase, "rank": training_args.process_index, "event": "train_begin", "resume": resume}
        ),
        flush=True,
    )
    trainer.train(resume_from_checkpoint=resume)
    expected_steps = 2 if args.phase == "save" else 4
    if trainer.state.global_step != expected_steps:
        raise AssertionError(f"Expected step {expected_steps}, got {trainer.state.global_step}")
    weights = snapshot()
    if weights is not None:
        save_file(weights, str(args.output_dir / f"{args.phase}-weights.safetensors"))
        report = {
            "phase": args.phase,
            "global_step": trainer.state.global_step,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "checkpoint_files": sorted(
                str(path.relative_to(args.output_dir)) for path in args.output_dir.glob("checkpoint-*/*")
            ),
        }
        (args.output_dir / f"{args.phase}-result.json").write_text(json.dumps(report, indent=2) + "\n")
    trainer.accelerator.wait_for_everyone()
    print(json.dumps({"phase": args.phase, "rank": training_args.process_index, "event": "complete"}), flush=True)


def main():
    args = parse_args()
    faulthandler.enable()
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    if not args.cpu and (not torch.cuda.is_available() or torch.cuda.device_count() < args.processes):
        raise SystemExit(
            f"This experiment requires {args.processes} CUDA GPUs; found {torch.cuda.device_count()}. Use the explicit plain CPU control only for harness validation."
        )
    if args.phase == "all":
        run_experiment(args)
    else:
        from torch.distributed.elastic.multiprocessing.errors import record

        faulthandler.dump_traceback_later(60, repeat=True)
        try:
            record(run_worker)(args)
        finally:
            faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
