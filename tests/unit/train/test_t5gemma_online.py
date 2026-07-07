import torch
from datasets import Dataset

import speculators.train.t5gemma_online as online


class _FakeExtractor:
    def extract_stacked(self, encoder_input_ids, decoder_input_ids):
        assert encoder_input_ids.numel() > 0
        seq_len = decoder_input_ids.shape[0]
        auxiliary = torch.ones(seq_len, 3)
        final = 2 * torch.ones(seq_len, 3)
        return torch.stack([auxiliary, final], dim=1)


def test_online_dataset_generates_states_without_cache(tmp_path, monkeypatch):
    data_path = tmp_path / "data"
    Dataset.from_dict(
        {
            "encoder_input_ids": [[10, 11]],
            "input_ids": [[2, 20, 21]],
            "loss_mask": [[0, 1, 1]],
            "seq_len": [3],
        }
    ).save_to_disk(data_path)
    monkeypatch.setattr(online, "_shared_extractor", _FakeExtractor())
    monkeypatch.setattr(
        online,
        "_extractor_settings",
        ("test", (0,), torch.float32),
    )

    dataset = online.T5GemmaOnlineDataset(
        max_len=8,
        datapath=data_path,
        split_ratio=1.0,
        hidden_states_dtype=torch.float32,
    )
    item = dataset[0]

    assert item is not None
    assert item["hidden_states"].shape == (3, 3)
    assert item["verifier_last_hidden_states"].shape == (3, 3)
    assert torch.equal(item["input_ids"], torch.tensor([2, 20, 21]))
    assert torch.equal(item["position_ids"], torch.arange(3))
    assert not (data_path / "hidden_states").exists()
