"""Evaluate a trained T5Gemma DFlash checkpoint on the validation split."""

import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as generic_train
from train_t5gemma import create_t5gemma_transformer_layer_config

from speculators.model import SpeculatorModel
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


def main():
    checkpoint = sys.argv[1]
    data_path = sys.argv[2] if len(sys.argv) > 2 else "./dflash-output/t5gemma-5k"

    generic_train.create_transformer_layer_config = create_t5gemma_transformer_layer_config

    logger.info("Loading checkpoint %s", checkpoint)
    model = SpeculatorModel.from_pretrained(checkpoint)
    model.to(torch.bfloat16).to("cuda")
    model.eval()

    dtype = torch.bfloat16
    target_layer_ids = model.config.aux_hidden_state_layer_ids
    configure_online_extractor(
        model.config.speculators_config.verifier.name_or_path,
        target_layer_ids,
        dtype,
    )

    dataset = T5GemmaOnlineDataset(
        max_len=2048,
        datapath=data_path,
        hidden_states_dtype=dtype,
        split_ratio=-0.1,
    )
    batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=2048,
        lengths=dataset.approx_lengths,
        num_replicas=1,
        rank=0,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=create_collate_fn(
            2048,
            model.config.transformer_layer_config.hidden_size,
            num_target_layers=len(target_layer_ids),
            dtype=dtype,
        ),
    )

    total_loss = 0.0
    total_acc_sum = 0.0
    total_acc_total = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            gpu_batch = {
                k: v.to("cuda", non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            _, loss, metrics = model(**gpu_batch)
            total_loss += loss.item()
            total_acc_sum += metrics["full_acc_sum"].item()
            total_acc_total += metrics["full_acc_total"].item()
            num_batches += 1
            if num_batches % 10 == 0:
                logger.info("Processed %d batches", num_batches)

    logger.info(
        "Validation: loss=%.4f, full_acc=%.4f over %d batches",
        total_loss / num_batches,
        total_acc_sum / (total_acc_total + 1e-8),
        num_batches,
    )
    close_online_extractor()


if __name__ == "__main__":
    main()
