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
"""Real distributed checkpoint experiments for Transformers #50549 / PEFT #4132."""

import sys

from parameterized import parameterized

from transformers.testing_utils import (
    TestCasePlus,
    execute_subprocess_async,
    require_deepspeed,
    require_peft,
    require_torch_multi_gpu,
    slow,
)


@slow
@require_peft
@require_torch_multi_gpu
class TestFSDPPeftResume(TestCasePlus):
    def run_experiment(self, scenario, *extra_args):
        command = [
            sys.executable,
            str(self.repo_root_dir / "scripts" / "reproduce_fsdp_peft_resume.py"),
            "--output-dir",
            self.get_auto_remove_tmp_dir(),
            "--scenario",
            scenario,
            *extra_args,
        ]
        execute_subprocess_async(command, env=self.get_env(), timeout=900)

    @parameterized.expand(
        [
            ("full_fsdp1", 1, "FULL_STATE_DICT"),
            ("sharded_fsdp1", 1, "SHARDED_STATE_DICT"),
            ("sharded_fsdp2", 2, "SHARDED_STATE_DICT"),
        ]
    )
    def test_resume(self, name, version, state_dict_type):
        self.run_experiment("resume", "--fsdp-version", str(version), "--state-dict-type", state_dict_type)

    @parameterized.expand(
        [
            ("full_fsdp1", 1, "FULL_STATE_DICT"),
            ("sharded_fsdp1", 1, "SHARDED_STATE_DICT"),
            ("sharded_fsdp2", 2, "SHARDED_STATE_DICT"),
        ]
    )
    def test_load_best_model(self, name, version, state_dict_type):
        self.run_experiment("best", "--fsdp-version", str(version), "--state-dict-type", state_dict_type)

    def test_named_adapters(self):
        self.run_experiment(
            "all", "--state-dict-type", "FULL_STATE_DICT", "--adapter-name", "trainable", "--extra-adapter"
        )

    def test_full_parameter_fsdp_control(self):
        self.run_experiment("resume", "--no-peft")

    def test_single_gpu_peft_control(self):
        self.run_experiment("resume", "--backend", "plain", "--processes", "1")

    @require_deepspeed
    def test_zero3_peft_control(self):
        self.run_experiment("resume", "--backend", "deepspeed")
