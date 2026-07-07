"""Train dFlash while generating T5Gemma decoder states in-process."""

import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as generic_train
from train_t5gemma import create_t5gemma_transformer_layer_config

from speculators.models.utils import resolve_target_layer_ids
from speculators.train.data import create_collate_fn
from speculators.train.distributed import get_dp_rank, get_dp_size
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.noise_transforms import AddUniformNoise
from speculators.train.t5gemma_online import (
    T5GemmaOnlineTokenDataset,
    close_online_extractor,
    configure_online_extractor,
    _get_online_extractor,
)

logger = logging.getLogger(__name__)


def setup_online_dataloader(
    dataset,
    hidden_size: int,
    num_workers: int = 0,  # noqa: ARG001
    num_target_layers: int = 3,
    prefetch_factor: int = 1,  # noqa: ARG001
    preprocess=None,
) -> DataLoader:
    """Build a same-process loader so the verifier is never copied to workers."""
    batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=generic_train.args.total_seq_len,
        lengths=dataset.approx_lengths,
        num_replicas=get_dp_size(),
        rank=get_dp_rank(),
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=create_t5gemma_online_collate_fn(
            generic_train.args.total_seq_len,
            hidden_size,
            num_target_layers=num_target_layers,
            dtype=dataset.hidden_states_dtype,
            transform=dataset.transform,
            preprocess=preprocess,
        ),
    )


def create_t5gemma_online_collate_fn(
    max_len: int,
    hidden_size: int,
    num_target_layers: int = 3,
    dtype: torch.dtype = torch.bfloat16,
    transform=None,
    preprocess=None,
):
    """Create a collate function that batches verifier hidden extraction."""
    base_collate = create_collate_fn(
        max_len,
        hidden_size,
        num_target_layers=num_target_layers,
        dtype=dtype,
        preprocess=preprocess,
    )

    def collate_fn(batch):
        raw_batch = [sample for sample in batch if sample is not None]
        if not raw_batch:
            return base_collate([])

        stacked_batch = _get_online_extractor().extract_stacked_batch(
            [sample["encoder_input_ids"] for sample in raw_batch],
            [sample["input_ids"] for sample in raw_batch],
        )

        processed = []
        for sample, stacked in zip(raw_batch, stacked_batch, strict=True):
            input_ids = sample["input_ids"].long()
            seq_len = input_ids.shape[0]
            item = {
                "hidden_states": stacked[:, :-1].flatten(1).to(dtype),
                "input_ids": input_ids,
                "verifier_last_hidden_states": stacked[:, -1].to(dtype),
                "loss_mask": sample["loss_mask"],
                "lengths": torch.tensor([seq_len], dtype=torch.long),
                "position_ids": torch.arange(seq_len, dtype=torch.long),
            }
            if transform:
                item = transform(item)
            processed.append(item)

        return base_collate(processed)

    return collate_fn


def create_online_train_val_loaders(
    *,
    data_path: str,
    train_data_ratio: float,
    total_seq_len: int,
    hidden_states_dtype: torch.dtype,
    noise_std: float,
    legacy_data: bool,
    hidden_states_path: str | None,  # noqa: ARG001
    vllm_endpoint: str,  # noqa: ARG001
    on_missing,  # noqa: ARG001
    on_generate,  # noqa: ARG001
    verifier_name_or_path: str,
    request_timeout: float | None,  # noqa: ARG001
    max_retries: int,  # noqa: ARG001
    hidden_size: int,
    num_target_layers: int,
    num_workers: int,
    prefetch_factor: int,
    preprocess=None,
):
    """Create online T5Gemma loaders.

    Hidden states are generated in-process through Transformers, so this path
    must not instantiate the generic ArrowDataset that expects cached hidden
    state files or a vLLM hidden-state endpoint.
    """
    if legacy_data:
        raise ValueError("Online T5Gemma training does not support --legacy-data")
    if not (0.0 < train_data_ratio < 1.0):
        raise ValueError(f"train_data_ratio must be in (0, 1), got {train_data_ratio}")

    train_dataset = T5GemmaOnlineTokenDataset(
        datapath=data_path,
        max_len=total_seq_len,
        transform=AddUniformNoise(std=noise_std),
        split_ratio=train_data_ratio,
        hidden_states_dtype=hidden_states_dtype,
        model=verifier_name_or_path,
    )
    val_dataset = T5GemmaOnlineTokenDataset(
        datapath=data_path,
        max_len=total_seq_len,
        split_ratio=train_data_ratio - 1.0,
        hidden_states_dtype=hidden_states_dtype,
        model=verifier_name_or_path,
    )

    return (
        setup_online_dataloader(
            train_dataset,
            hidden_size=hidden_size,
            num_workers=num_workers,
            num_target_layers=num_target_layers,
            prefetch_factor=prefetch_factor,
            preprocess=preprocess,
        ),
        setup_online_dataloader(
            val_dataset,
            hidden_size=hidden_size,
            num_workers=num_workers,
            num_target_layers=num_target_layers,
            prefetch_factor=prefetch_factor,
            preprocess=preprocess,
        ),
    )


def main() -> None:
    generic_train.create_transformer_layer_config = (
        create_t5gemma_transformer_layer_config
    )
    generic_train.create_train_val_loaders = create_online_train_val_loaders

    args = generic_train.parse_args()
    if args.speculator_type != "dflash":
        raise ValueError("Online T5Gemma training only supports dflash")
    if not hasattr(torch, args.hidden_states_dtype):
        raise ValueError(f"Unknown torch dtype: {args.hidden_states_dtype}")
    args.target_layer_ids = resolve_target_layer_ids(
        args.target_layer_ids, args.verifier_name_or_path
    )
    args.num_workers = 0
    generic_train.args = args
    configure_online_extractor(
        args.verifier_name_or_path,
        args.target_layer_ids,
        getattr(torch, args.hidden_states_dtype),
    )

    logger.info(
        "Online mode: verifier and drafter share one GPU; hidden states are not cached"
    )
    try:
        generic_train.main(args)
    finally:
        close_online_extractor()


if __name__ == "__main__":
    main()
