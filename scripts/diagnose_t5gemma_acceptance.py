"""Compare training accuracy with real full-vocabulary greedy acceptance.

This diagnostic intentionally evaluates the first speculative position.  For
DSpark that position only depends on the verifier-produced anchor token, so no
teacher-forcing mismatch is involved.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_t5gemma_online import create_t5gemma_online_collate_fn

from speculators.model import SpeculatorModel
from speculators.models.dspark import DSparkDraftModel
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.t5gemma_online import (
    T5GemmaOnlineTokenDataset,
    close_online_extractor,
    configure_online_extractor,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("data_path")
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--total-seq-len", type=int, default=2048)
    parser.add_argument("--max-anchors", type=int, default=256)
    parser.add_argument("--logit-chunk-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-data-ratio", type=float, default=0.9)
    return parser.parse_args()


def _full_vocab_argmax(
    hidden_states: torch.Tensor,
    output_weight: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Compute top-1 IDs without materializing all position/vocab logits."""
    result = []
    for start in range(0, hidden_states.shape[0], chunk_size):
        logits = F.linear(hidden_states[start : start + chunk_size], output_weight)
        result.append(logits.argmax(dim=-1))
    return torch.cat(result)


def _apply_dspark_markov(
    model: DSparkDraftModel,
    hidden: torch.Tensor,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    anchored_block_indices: torch.Tensor,
    num_blocks: int,
) -> torch.Tensor:
    if model.markov_head is None:
        return logits
    block = model.block_size
    block_tokens = input_ids[0, anchored_block_indices].view(num_blocks, block)
    prev_token_ids = torch.cat([block_tokens[:, :1], block_tokens[:, :-1]], dim=1)
    hidden_blocks = hidden.view(num_blocks, block, -1)
    prev_emb = model.markov_head.prev_embeddings(prev_token_ids)
    bias = model.markov_head.block_bias(
        prev_token_ids=prev_token_ids,
        hidden_states=hidden_blocks,
        prev_emb=prev_emb,
    )
    return (logits.view(num_blocks, block, -1) + bias).view_as(logits)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.max_batches <= 0 or args.max_anchors <= 0:
        raise ValueError("--max-batches and --max-anchors must be positive")

    torch.manual_seed(args.seed)
    model = SpeculatorModel.from_pretrained(args.checkpoint)
    if not isinstance(model, DSparkDraftModel):
        raise TypeError("This diagnostic currently expects a DSpark checkpoint")
    model.to(device="cuda", dtype=torch.bfloat16).eval()

    target_layer_ids = model.config.aux_hidden_state_layer_ids
    verifier_name = model.config.speculators_config.verifier.name_or_path
    configure_online_extractor(verifier_name, target_layer_ids, torch.bfloat16)

    dataset = T5GemmaOnlineTokenDataset(
        max_len=args.total_seq_len,
        datapath=args.data_path,
        split_ratio=args.train_data_ratio - 1.0,
        hidden_states_dtype=torch.bfloat16,
    )
    sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=args.total_seq_len,
        lengths=dataset.approx_lengths,
        num_replicas=1,
        rank=0,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=create_t5gemma_online_collate_fn(
            args.total_seq_len,
            model.config.transformer_layer_config.hidden_size,
            num_target_layers=len(target_layer_ids),
            dtype=torch.bfloat16,
        ),
    )

    totals = {
        "count": 0,
        "restricted_correct": 0,
        "full_vocab_covered": 0,
        "full_correct": 0,
        "restricted_target_matches_full": 0,
    }
    try:
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.max_batches:
                break
            gpu = {
                key: value.to("cuda", non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            torch.manual_seed(args.seed + batch_idx)
            hidden, logits, targets, loss_mask, indices = model._backbone_forward(
                gpu["hidden_states"],
                gpu["input_ids"],
                gpu["loss_mask"],
                gpu["verifier_last_hidden_states"],
                gpu["document_ids"],
                gpu.get("position_ids"),
                max_anchors=args.max_anchors,
            )
            logits = _apply_dspark_markov(
                model,
                hidden,
                logits,
                gpu["input_ids"],
                indices,
                args.max_anchors,
            )

            positions = torch.arange(logits.shape[1], device="cuda")
            select = (positions % model.block_size == 1) & loss_mask[0].bool()
            selected_indices = indices[select]
            draft_ids = logits[0, select].argmax(dim=-1)
            restricted_target_ids = targets[0, select].argmax(dim=-1)
            target_ids = draft_ids + model.d2t[draft_ids]
            restricted_target_target_ids = (
                restricted_target_ids + model.d2t[restricted_target_ids]
            )

            # Position i is predicted by verifier hidden state i - 1. Position 1
            # in every anchored block therefore uses the anchor hidden state.
            verifier_hidden = gpu["verifier_last_hidden_states"][
                0, selected_indices - 1
            ]
            full_target_ids = _full_vocab_argmax(
                verifier_hidden,
                model.embed_tokens.weight,
                args.logit_chunk_size,
            )

            count = int(full_target_ids.numel())
            totals["count"] += count
            totals["restricted_correct"] += int(
                (draft_ids == restricted_target_ids).sum().item()
            )
            totals["full_vocab_covered"] += int(
                model.t2d[full_target_ids].sum().item()
            )
            totals["full_correct"] += int((target_ids == full_target_ids).sum().item())
            totals["restricted_target_matches_full"] += int(
                (restricted_target_target_ids == full_target_ids).sum().item()
            )
            logger.info("Processed batch %d: %d valid position-1 anchors", batch_idx, count)
    finally:
        close_online_extractor()

    count = totals["count"]
    if count == 0:
        raise RuntimeError("No valid position-1 anchors were evaluated")
    print(f"position_1_count: {count}")
    print(f"restricted_position_1_accuracy: {totals['restricted_correct'] / count:.6f}")
    print(f"full_vocab_top1_coverage: {totals['full_vocab_covered'] / count:.6f}")
    print(
        "restricted_target_matches_full_top1: "
        f"{totals['restricted_target_matches_full'] / count:.6f}"
    )
    print(f"real_full_vocab_position_1_acceptance: {totals['full_correct'] / count:.6f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
