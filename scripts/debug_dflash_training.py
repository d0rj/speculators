"""Debug one forward/backward pass of online T5Gemma dFlash training."""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as generic_train
from train_t5gemma import create_t5gemma_transformer_layer_config

from speculators.model import SpeculatorModel
from speculators.models.metrics import ce_loss
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def setup_online_dataloader(
    dataset,
    world_size: int,
    local_rank: int,
    hidden_size: int,
    num_target_layers: int = 3,
):
    batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=2048,
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
            2048,
            hidden_size,
            num_target_layers=num_target_layers,
            dtype=torch.bfloat16,
        ),
    )


def main():
    generic_train.create_transformer_layer_config = create_t5gemma_transformer_layer_config
    generic_train.ArrowDataset = T5GemmaOnlineDataset
    generic_train.setup_dataloader = setup_online_dataloader

    model_name = "google/t5gemma-2-1b-1b"
    data_path = "./dflash-output/t5gemma-online-200k"
    target_layer_ids = [2, 13, 23]
    draft_vocab_size = 32000
    num_layers = 2

    configure_online_extractor(model_name, target_layer_ids, torch.bfloat16)

    transformer_layer_config = create_t5gemma_transformer_layer_config(
        verifier_name_or_path=model_name,
        num_layers=num_layers,
        draft_arch="llama",
        hidden_act="silu",
        sliding_window=2048,
        sliding_window_indices=[],
    )

    d2t = torch.from_numpy(np.load(Path(data_path) / "d2t.npy"))
    t2d = torch.from_numpy(np.load(Path(data_path) / "t2d.npy"))

    model = SpeculatorModel.registry["dflash"].from_training_args(
        verifier_config=transformer_layer_config,
        verifier_name_or_path=model_name,
        draft_vocab_size=draft_vocab_size,
        speculator_type="dflash",
        draft_arch="llama",
        draft_hidden_act="silu",
        draft_attn_impl="eager",
        block_size=8,
        max_anchors=128,
        target_layer_ids=target_layer_ids,
        mask_token_id=4,
        sliding_window_non_causal=False,
        loss_fn="ce",
        t2d=t2d,
        d2t=d2t,
    )
    model.to(torch.bfloat16).to("cuda")
    model.train()

    logger.info("Model created on %s", next(model.parameters()).device)

    dataset = T5GemmaOnlineDataset(
        max_len=2048,
        datapath=data_path,
        hidden_states_dtype=torch.bfloat16,
        split_ratio=1.0,
    )
    loader = setup_online_dataloader(
        dataset, 1, 0, transformer_layer_config.hidden_size, num_target_layers=len(target_layer_ids)
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4,
        weight_decay=0.01,
    )

    history = []
    for step, batch in enumerate(loader):
        if step >= 100:
            break
        gpu_batch = {
            k: v.to("cuda", non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        draft_tokens, loss, metrics = model(**gpu_batch, loss_fn=ce_loss)
        m = {k: v.item() for k, v in metrics.items()}
        full_total = m.get("full_acc_total", 1)
        full_acc = m["full_acc_sum"] / full_total if full_total > 0 else 0.0

        optimizer.zero_grad()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        history.append((step, loss.item(), full_acc, total_norm.item()))
        if step % 10 == 0:
            logger.info(
                "step %d loss=%.4f full_acc=%.4f grad_norm=%.4f",
                step, loss.item(), full_acc, total_norm.item()
            )

    logger.info("Final 10 steps:")
    for step, loss, acc, gnorm in history[-10:]:
        logger.info("step %d loss=%.4f full_acc=%.4f grad_norm=%.4f", step, loss, acc, gnorm)

    close_online_extractor()


if __name__ == "__main__":
    main()
