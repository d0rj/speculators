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
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.t5gemma_online import (
    T5GemmaOnlineDataset,
    close_online_extractor,
    configure_online_extractor,
)

logger = logging.getLogger(__name__)


def setup_online_dataloader(
    dataset,
    world_size: int,
    local_rank: int,
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
        num_replicas=world_size,
        rank=local_rank,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=create_collate_fn(
            generic_train.args.total_seq_len,
            hidden_size,
            num_target_layers=num_target_layers,
            dtype=dataset.hidden_states_dtype,
            preprocess=preprocess,
        ),
    )


def main() -> None:
    generic_train.create_transformer_layer_config = (
        create_t5gemma_transformer_layer_config
    )
    generic_train.ArrowDataset = T5GemmaOnlineDataset
    generic_train.setup_dataloader = setup_online_dataloader

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
