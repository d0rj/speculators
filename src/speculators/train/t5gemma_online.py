"""In-process T5Gemma hidden-state generation for online training."""

import logging
import os
from os import PathLike
from typing import Literal

import torch
from datasets import load_from_disk
from transformers import AutoConfig, AutoModelForSeq2SeqLM

from speculators.models.utils import get_verifier_text_config
from speculators.train.data import BaseDataset
from speculators.train.noise_transforms import TransformTensors

logger = logging.getLogger(__name__)


def _output_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"Unsupported hook output: {type(output)}")


class T5GemmaHiddenStateExtractor:
    """Run the verifier and capture selected normalized decoder states."""

    def __init__(
        self,
        model_name_or_path: str,
        target_layer_ids: list[int],
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for T5Gemma hidden-state extraction")

        self.model_name_or_path = model_name_or_path
        self.target_layer_ids = target_layer_ids
        self.dtype = dtype
        self.device = torch.device(device)
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.decoder_config = get_verifier_text_config(self.config)
        self._validate_target_layers()

        logger.info("Loading online verifier %s", model_name_or_path)
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                dtype=dtype,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
            .eval()
            .to(self.device)
        )
        self.model.requires_grad_(False)
        self.decoder = self.model.model.decoder
        self._captures: dict[int | str, torch.Tensor] = {}
        self._handles = []
        self._num_extractions = 0
        self._register_hooks()

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            logger.info(
                "Verifier loaded; combined GPU allocated/reserved %.2f/%.2f GiB",
                torch.cuda.memory_allocated(self.device) / 2**30,
                torch.cuda.memory_reserved(self.device) / 2**30,
            )

    def _validate_target_layers(self) -> None:
        invalid = [
            idx
            for idx in self.target_layer_ids
            if idx < 0 or idx >= self.decoder_config.num_hidden_layers
        ]
        if invalid:
            raise ValueError(f"Invalid decoder layer ids: {invalid}")
        if len(set(self.target_layer_ids)) != len(self.target_layer_ids):
            raise ValueError("Target layer ids must be unique")

    def _register_hooks(self) -> None:
        for layer_id in self.target_layer_ids:
            layer = self.decoder.layers[layer_id]

            def hook(_module, _inputs, output, layer_id=layer_id):
                self._captures[layer_id] = _output_tensor(output).detach()

            # T5Gemma2 applies RMSNorm *inside* the residual path, so the layer
            # output is an unnormalized residual sum.  Capture the normalized
            # sub-layer output before the residual add when possible.
            if hasattr(layer, "post_feedforward_layernorm"):
                submodule = layer.post_feedforward_layernorm
            else:
                submodule = layer
            self._handles.append(submodule.register_forward_hook(hook))

        def decoder_hook(_module, _inputs, output):
            self._captures["last"] = _output_tensor(output).detach()

        self._handles.append(self.decoder.register_forward_hook(decoder_hook))

    @torch.inference_mode()
    def _extract_stacked_padded(
        self,
        encoder_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        decoder_lengths: list[int],
    ) -> list[torch.Tensor]:
        encoder_ids = encoder_ids.long().to(self.device)
        encoder_attention_mask = encoder_attention_mask.long().to(self.device)
        decoder_ids = decoder_ids.long().to(self.device)
        decoder_attention_mask = decoder_attention_mask.long().to(self.device)
        self._captures.clear()

        # Call the base encoder-decoder model directly. The LM wrapper would
        # materialize [decoder_length, 262k] logits that online training does not use.
        self.model.model(
            input_ids=encoder_ids,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        missing = [
            layer_id
            for layer_id in self.target_layer_ids
            if layer_id not in self._captures
        ]
        if missing or "last" not in self._captures:
            raise RuntimeError(f"Missing hook outputs: layers={missing}")

        # Captured tensors have shape [batch, decoder_len, hidden_size].
        # Stack to [batch, decoder_len, num_target_layers + 1, hidden_size].
        hidden_states = torch.stack(
            [self._captures[layer_id] for layer_id in self.target_layer_ids]
            + [self._captures["last"]],
            dim=2,
        ).to(device="cpu", dtype=self.dtype)
        self._captures.clear()
        self._num_extractions += 1
        if self.device.type == "cuda" and self._num_extractions % 10 == 0:
            logger.info(
                "Online extraction %d; batch=%d; peak GPU allocated/reserved %.2f/%.2f GiB",
                self._num_extractions,
                encoder_ids.shape[0],
                torch.cuda.max_memory_allocated(self.device) / 2**30,
                torch.cuda.max_memory_reserved(self.device) / 2**30,
            )
        return [
            hidden_states[idx, :decoder_length].contiguous()
            for idx, decoder_length in enumerate(decoder_lengths)
        ]

    @staticmethod
    def _pad_batch(
        sequences: list[torch.Tensor],
        pad_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        lengths = [int(seq.shape[0]) for seq in sequences]
        max_len = max(lengths)
        ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
        for idx, seq in enumerate(sequences):
            length = lengths[idx]
            ids[idx, :length] = seq.long()
            mask[idx, :length] = 1
        return ids, mask, lengths

    def _pad_token_id(self) -> int:
        pad_token_id = (
            getattr(self.config, "pad_token_id", None)
            or getattr(self.decoder_config, "pad_token_id", None)
            or 0
        )
        if isinstance(pad_token_id, list):
            pad_token_id = pad_token_id[0]
        return int(pad_token_id)

    @torch.inference_mode()
    def extract_stacked_batch(
        self,
        encoder_input_ids: list[torch.Tensor],
        decoder_input_ids: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Extract verifier hidden states for a batch of variable-length samples."""
        if len(encoder_input_ids) != len(decoder_input_ids):
            raise ValueError("encoder_input_ids and decoder_input_ids size mismatch")
        if not encoder_input_ids:
            return []

        max_batch_size = int(os.getenv("T5GEMMA_ONLINE_EXTRACTOR_BATCH_SIZE", "4"))
        if max_batch_size <= 0:
            max_batch_size = len(encoder_input_ids)

        pad_token_id = self._pad_token_id()
        outputs: list[torch.Tensor] = []
        for start in range(0, len(encoder_input_ids), max_batch_size):
            end = min(start + max_batch_size, len(encoder_input_ids))
            enc_ids, enc_mask, _enc_lengths = self._pad_batch(
                encoder_input_ids[start:end], pad_token_id
            )
            dec_ids, dec_mask, dec_lengths = self._pad_batch(
                decoder_input_ids[start:end], pad_token_id
            )
            outputs.extend(
                self._extract_stacked_padded(
                    enc_ids,
                    enc_mask,
                    dec_ids,
                    dec_mask,
                    dec_lengths,
                )
            )
        return outputs

    @torch.inference_mode()
    def extract_stacked(
        self,
        encoder_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.extract_stacked_batch(
            [encoder_input_ids],
            [decoder_input_ids],
        )[0]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


_extractor_settings: tuple[str, tuple[int, ...], torch.dtype] | None = None
_shared_extractor: T5GemmaHiddenStateExtractor | None = None


def configure_online_extractor(
    model_name_or_path: str,
    target_layer_ids: list[int],
    dtype: torch.dtype,
) -> None:
    global _extractor_settings, _shared_extractor  # noqa: PLW0603
    settings = (model_name_or_path, tuple(target_layer_ids), dtype)
    if _extractor_settings != settings:
        if _shared_extractor is not None:
            _shared_extractor.close()
        _shared_extractor = None
        _extractor_settings = settings


def _get_online_extractor() -> T5GemmaHiddenStateExtractor:
    global _shared_extractor  # noqa: PLW0603
    if _extractor_settings is None:
        raise RuntimeError("Online T5Gemma extractor has not been configured")
    if _shared_extractor is None:
        model_name_or_path, target_layer_ids, dtype = _extractor_settings
        _shared_extractor = T5GemmaHiddenStateExtractor(
            model_name_or_path,
            list(target_layer_ids),
            dtype=dtype,
        )
    return _shared_extractor


def close_online_extractor() -> None:
    global _shared_extractor  # noqa: PLW0603
    if _shared_extractor is not None:
        _shared_extractor.close()
        _shared_extractor = None


class T5GemmaOnlineDataset(BaseDataset):
    """Arrow dataset that computes verifier hidden states on every access."""

    def __init__(
        self,
        max_len: int,
        datapath: str | PathLike,
        hidden_states_path: str | PathLike | None = None,  # noqa: ARG002
        vllm_endpoint: str = "http://localhost:8000/v1",  # noqa: ARG002
        on_missing: Literal["generate", "skip", "warn", "raise"] = "generate",  # noqa: ARG002
        on_generate: Literal["cache", "delete"] = "delete",  # noqa: ARG002
        split_ratio: float = 1.0,
        transform: TransformTensors | None = None,
        hidden_states_dtype=torch.bfloat16,
        model: str | None = None,  # noqa: ARG002
        request_timeout: float | None = None,  # noqa: ARG002
        max_retries: int = 0,  # noqa: ARG002
    ) -> None:
        self.data = load_from_disk(datapath)
        if split_ratio == 1.0:
            pass
        elif 1.0 > split_ratio > 0:
            self.data = self.data.select(range(int(len(self.data) * split_ratio)))
        elif -1.0 < split_ratio < 0:
            split_idx = int(len(self.data) * (1.0 + split_ratio))
            self.data = self.data.select(range(split_idx, len(self.data)))
        else:
            raise ValueError("split_ratio must be in range (-1.0, 1.0] excluding 0.0")
        super().__init__(max_len, transform, hidden_states_dtype)

    def __len__(self) -> int:
        return len(self.data)

    def _compute_approx_lengths(self) -> list[int]:
        return list(self.data.with_format(None)["seq_len"])

    def _get_raw_data(self, index):
        item = self.data.with_format("torch")[index]
        stacked = _get_online_extractor().extract_stacked(
            item["encoder_input_ids"], item["input_ids"]
        )
        return {
            "hidden_states": stacked[:, :-1].flatten(1),
            "input_ids": item["input_ids"].long(),
            "verifier_last_hidden_states": stacked[:, -1],
            "loss_mask": item["loss_mask"],
        }


class T5GemmaOnlineTokenDataset(torch.utils.data.Dataset):
    """Arrow dataset that returns tokens only.

    Hidden states are generated later in the collate function so the frozen
    verifier can process multiple samples in one forward pass.
    """

    def __init__(
        self,
        max_len: int,
        datapath: str | PathLike,
        split_ratio: float = 1.0,
        transform: TransformTensors | None = None,
        hidden_states_dtype=torch.bfloat16,
        model: str | None = None,  # noqa: ARG002
    ) -> None:
        self.max_len = max_len
        self.transform = transform
        self.hidden_states_dtype = hidden_states_dtype
        self.data = load_from_disk(datapath)
        if split_ratio == 1.0:
            pass
        elif 1.0 > split_ratio > 0:
            self.data = self.data.select(range(int(len(self.data) * split_ratio)))
        elif -1.0 < split_ratio < 0:
            split_idx = int(len(self.data) * (1.0 + split_ratio))
            self.data = self.data.select(range(split_idx, len(self.data)))
        else:
            raise ValueError("split_ratio must be in range (-1.0, 1.0] excluding 0.0")
        self.approx_lengths = self._compute_approx_lengths()

    def __len__(self) -> int:
        return len(self.data)

    def _compute_approx_lengths(self) -> list[int]:
        return list(self.data.with_format(None)["seq_len"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.data.with_format("torch")[index]
        return {
            "encoder_input_ids": item["encoder_input_ids"].long(),
            "input_ids": item["input_ids"].long(),
            "loss_mask": item["loss_mask"],
        }
