# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch ViT model."""

import copy
import tempfile
import unittest
from functools import cached_property

from transformers import ViTConfig
from transformers.testing_utils import (
    require_accelerate,
    require_torch,
    require_torch_accelerator,
    require_torch_fp16,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import ViTForImageClassification, ViTForMaskedImageModeling, ViTModel


if is_vision_available():
    from PIL import Image

    from transformers import ViTImageProcessorPil


class ViTModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_sequence_label_size=10,
        initializer_range=0.02,
        scope=None,
        encoder_stride=2,
        mask_ratio=0.5,
        attn_implementation="eager",
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.scope = scope
        self.encoder_stride = encoder_stride
        self.attn_implementation = attn_implementation

        # in ViT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1
        self.mask_ratio = mask_ratio
        self.num_masks = int(mask_ratio * self.seq_length)
        self.mask_length = num_patches

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return ViTConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
            encoder_stride=self.encoder_stride,
            attn_implementation=self.attn_implementation,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = ViTModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_masked_image_modeling(self, config, pixel_values, labels):
        model = ViTForMaskedImageModeling(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(
            result.reconstruction.shape, (self.batch_size, self.num_channels, self.image_size, self.image_size)
        )

        # test greyscale images
        config.num_channels = 1
        model = ViTForMaskedImageModeling(config)
        model.to(torch_device)
        model.eval()

        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values)
        self.parent.assertEqual(result.reconstruction.shape, (self.batch_size, 1, self.image_size, self.image_size))

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.type_sequence_label_size
        model = ViTForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

        # test greyscale images
        config.num_channels = 1
        model = ViTForImageClassification(config)
        model.to(torch_device)
        model.eval()

        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values,
            labels,
        ) = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class ViTModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as ViT does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (
            ViTModel,
            ViTForImageClassification,
            ViTForMaskedImageModeling,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {"image-feature-extraction": ViTModel, "image-classification": ViTForImageClassification}
        if is_torch_available()
        else {}
    )

    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = ViTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ViTConfig, has_text_modality=False, hidden_size=32)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="ViT does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_image_modeling(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_image_modeling(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    def test_adapt_input_channels(self):
        config = self.model_tester.get_config()

        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config)).to(torch_device)
            model.eval()
            base_model = getattr(model, model.base_model_prefix, model)
            for channels, mode in ((4, "luminance"), (12, "variance"), (1, "luminance")):
                with self.subTest(model=model_class.__name__, channels=channels):
                    projection = model.adapt_input_channels(channels, mode=mode)
                    self.assertIs(projection, base_model.embeddings.patch_embeddings.projection)
                    self.assertEqual(model.config.num_channels, channels)
                    self.assertEqual(base_model.config.num_channels, channels)
                    self.assertEqual(base_model.embeddings.patch_embeddings.num_channels, channels)
                    inputs = torch.randn(2, channels, config.image_size, config.image_size, device=torch_device)
                    self.assertIsNotNone(model(inputs))
                    self.assertIs(model.adapt_input_channels(channels), projection)

            model.train()
            model.adapt_input_channels(4)
            inputs = torch.randn(2, 4, config.image_size, config.image_size, device=torch_device)
            outputs = model(inputs)
            out_tensor = outputs[0]
            loss = (out_tensor * torch.randn_like(out_tensor)).sum()
            loss.backward()
            weight_grad = base_model.embeddings.patch_embeddings.projection.weight.grad
            self.assertIsNotNone(weight_grad)
            self.assertTrue(torch.isfinite(weight_grad).all())
            per_ch = weight_grad.abs().amax(dim=(0, 2, 3))
            self.assertTrue(per_ch.max() > 0)
            self.assertTrue((per_ch > 1e-3 * per_ch.max()).all())

    def test_adapt_input_channels_validation(self):
        model = ViTModel(self.model_tester.get_config())
        projection = model.embeddings.patch_embeddings.projection
        for channels, error in ((0, ValueError), (-3, ValueError), (True, TypeError), (2.5, TypeError)):
            with self.subTest(channels=channels), self.assertRaises(error):
                model.adapt_input_channels(channels)
        for channels in (1, model.config.num_channels, 4):
            with self.subTest(channels=channels), self.assertRaises(ValueError):
                model.adapt_input_channels(channels, mode="invalid_mode")
        self.assertIs(model.embeddings.patch_embeddings.projection, projection)
        self.assertEqual(model.config.num_channels, self.model_tester.num_channels)

    def test_adapt_input_channels_preserves_projection_attributes(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = ViTModel(self.model_tester.get_config()).to(device=torch_device, dtype=dtype).eval()
                old = model.embeddings.patch_embeddings.projection
                old.weight.requires_grad_(False)
                old.bias.requires_grad_(True)
                with torch.no_grad():
                    old.bias.fill_(0.25)
                projection = model.adapt_input_channels(4)
                self.assertEqual(projection.weight.dtype, dtype)
                self.assertEqual(projection.weight.device, old.weight.device)
                self.assertFalse(projection.weight.requires_grad)
                self.assertTrue(projection.bias.requires_grad)
                self.assertFalse(projection.training)
                torch.testing.assert_close(projection.bias, old.bias, rtol=0, atol=0)
                for name in ("stride", "padding", "dilation", "groups", "padding_mode"):
                    self.assertEqual(getattr(projection, name), getattr(old, name))

        model = ViTModel(self.model_tester.get_config())
        model.embeddings.patch_embeddings.projection.bias = None
        self.assertIsNone(model.adapt_input_channels(4).bias)

    def test_adapt_input_channels_constant(self):
        config = self.model_tester.get_config()
        model = ViTModel(copy.deepcopy(config))
        weight = model.embeddings.patch_embeddings.projection.weight.detach().clone()
        for channels in (1, 2, 4, 12):
            with self.subTest(channels=channels):
                adapted = ViTModel(copy.deepcopy(config))
                adapted.load_state_dict(model.state_dict())
                projection = adapted.adapt_input_channels(channels, mode="constant")
                expected = weight[:, torch.arange(channels) % config.num_channels]
                torch.testing.assert_close(projection.weight, expected, rtol=0, atol=0)

    def test_adapt_input_channels_energy_conservation(self):
        config = self.model_tester.get_config()
        model = ViTModel(copy.deepcopy(config))
        with torch.no_grad():
            model.embeddings.patch_embeddings.projection.weight.mul_(torch.tensor([1.5, 0.8, 0.3]).view(1, 3, 1, 1))
        for dtype, tolerance in ((torch.float32, 1e-5), (torch.float16, 2e-3), (torch.bfloat16, 2e-2)):
            for channels in (1, 2, 4, 12, 128):
                with self.subTest(dtype=dtype, channels=channels):
                    adapted = ViTModel(copy.deepcopy(config)).to(dtype)
                    adapted.load_state_dict(model.state_dict())
                    old_energy = adapted.embeddings.patch_embeddings.projection.weight.detach().float().square().sum()
                    projection = adapted.adapt_input_channels(channels, mode="variance")
                    energy = projection.weight.detach().float().square().sum()
                    self.assertTrue(torch.isfinite(energy))
                    torch.testing.assert_close(energy / old_energy, torch.ones_like(energy), rtol=tolerance, atol=0)

        model = ViTModel(copy.deepcopy(config)).half()
        with torch.no_grad():
            model.embeddings.patch_embeddings.projection.weight.fill_(1.0)
        old_energy = model.embeddings.patch_embeddings.projection.weight.detach().float().square().sum()
        projection = model.adapt_input_channels(1024, mode="variance")
        energy = projection.weight.detach().float().square().sum()
        torch.testing.assert_close(energy / old_energy, torch.ones_like(energy), rtol=2e-3, atol=0)

    def test_adapt_input_channels_luminance_equivalence(self):
        config = self.model_tester.get_config()
        gray = torch.randn(2, 1, config.image_size, config.image_size)
        for old_channels in (1, 3, 4, 12):
            config.num_channels = old_channels
            model = ViTModel(copy.deepcopy(config))
            with torch.no_grad():
                model.embeddings.patch_embeddings.projection.bias.fill_(0.25)
                expected = model.embeddings.patch_embeddings(gray.expand(-1, old_channels, -1, -1))
            for channels in (1, 2, 4, 12, 128):
                with self.subTest(old_channels=old_channels, channels=channels):
                    adapted = ViTModel(copy.deepcopy(config))
                    adapted.load_state_dict(model.state_dict())
                    adapted.adapt_input_channels(channels)
                    with torch.no_grad():
                        actual = adapted.embeddings.patch_embeddings(gray.expand(-1, channels, -1, -1))
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_adapt_input_channels_masked_image_modeling(self):
        config = self.model_tester.get_config()
        model = ViTForMaskedImageModeling(config).to(torch_device).eval()
        old_decoder = model.decoder[0]
        with torch.no_grad():
            old_decoder.bias.copy_(torch.arange(old_decoder.out_channels, device=torch_device))
        stride_squared = config.encoder_stride**2
        expected_weight = old_decoder.weight.detach().reshape(
            config.num_channels, stride_squared, config.hidden_size, 1, 1
        )
        expected_bias = old_decoder.bias.detach().reshape(config.num_channels, stride_squared)
        model.adapt_input_channels(4)
        decoder = model.decoder[0]
        self.assertFalse(decoder.training)
        torch.testing.assert_close(decoder.weight[: old_decoder.out_channels], old_decoder.weight, rtol=0, atol=0)
        torch.testing.assert_close(decoder.weight[-stride_squared:], expected_weight[0], rtol=0, atol=0)
        torch.testing.assert_close(decoder.bias[-stride_squared:], expected_bias[0], rtol=0, atol=0)
        pixels = torch.randn(2, 4, config.image_size, config.image_size, device=torch_device)
        num_patches = (config.image_size // config.patch_size) ** 2
        mask = torch.arange(num_patches, device=torch_device).remainder(2).bool().expand(2, -1)
        outputs = model(pixels, bool_masked_pos=mask)
        self.assertEqual(outputs.reconstruction.shape, pixels.shape)
        self.assertTrue(torch.isfinite(outputs.loss))
        outputs.loss.backward()
        self.assertTrue(torch.isfinite(decoder.weight.grad).all())
        self.assertGreater(decoder.weight.grad.abs().sum().item(), 0)

        weight = decoder.weight.detach().reshape(4, stride_squared, config.hidden_size, 1, 1).mean(dim=0)
        bias = decoder.bias.detach().reshape(4, stride_squared).mean(dim=0)
        model.adapt_input_channels(1)
        torch.testing.assert_close(model.decoder[0].weight, weight)
        torch.testing.assert_close(model.decoder[0].bias, bias)

    def test_adapt_input_channels_save_load_roundtrip(self):
        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            for channels in (1, 4):
                with self.subTest(model=model_class.__name__, channels=channels):
                    model = model_class(copy.deepcopy(config)).eval()
                    model.adapt_input_channels(channels)
                    pixels = torch.randn(2, channels, config.image_size, config.image_size)
                    with torch.no_grad():
                        expected = model(pixels)[0]
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        model.save_pretrained(tmp_dir)
                        loaded = model_class.from_pretrained(
                            tmp_dir, attn_implementation=model.config._attn_implementation
                        ).eval()
                    self.assertEqual(loaded.config.num_channels, channels)
                    for name, value in model.state_dict().items():
                        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
                    with torch.no_grad():
                        torch.testing.assert_close(loaded(pixels)[0], expected, rtol=0, atol=0)

    @slow
    def test_model_from_pretrained(self):
        model_name = "google/vit-base-patch16-224"
        model = ViTModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class ViTModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return ViTImageProcessorPil.from_pretrained("google/vit-base-patch16-224") if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([-0.2744, 0.8215, -0.0836]).to(torch_device)

        torch.testing.assert_close(outputs.logits[0, :3], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_interpolate_pos_encoding(self):
        # ViT models have an `interpolate_pos_encoding` argument in their forward method,
        # allowing to interpolate the pre-trained position embeddings in order to use
        # the model on higher resolutions. The DINO model by Facebook AI leverages this
        # to visualize self-attention on higher resolution images.
        model = ViTModel.from_pretrained("facebook/dino-vits8").to(torch_device)

        image_processor = ViTImageProcessorPil.from_pretrained("facebook/dino-vits8", size=480)
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(pixel_values, interpolate_pos_encoding=True)

        # verify the logits
        expected_shape = torch.Size((1, 3601, 384))
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)

        expected_slice = torch.tensor(
            [[4.2325, 4.3882, -6.6678], [4.5372, 1.8933, -6.7355], [4.4454, 0.8514, -5.8747]]
        ).to(torch_device)

        torch.testing.assert_close(outputs.last_hidden_state[0, :3, :3], expected_slice, rtol=1e-3, atol=1e-3)

    @slow
    @require_accelerate
    @require_torch_accelerator
    @require_torch_fp16
    def test_inference_fp16(self):
        r"""
        A small test to make sure that inference work in half precision without any problem.
        """
        model = ViTModel.from_pretrained("facebook/dino-vits8", dtype=torch.float16, device_map="auto")
        image_processor = self.default_image_processor

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(torch_device)

        # forward pass to make sure inference works in fp16
        with torch.no_grad():
            _ = model(pixel_values)
