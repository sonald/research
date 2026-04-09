from __future__ import annotations

import copy

import torch

from mini_megatrain.config import Config, DataConfig, ModelConfig, TrainingConfig
from mini_megatrain.engine import StreamingTrainer
from mini_megatrain.model import MiniTransformerLM


def _dense_step(model: MiniTransformerLM, batch: dict[str, torch.Tensor], lr: float) -> tuple[float, dict[str, torch.Tensor]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch["input_ids"])
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch["labels"][:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    loss.backward()
    optimizer.step()
    return loss.item(), {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def test_streaming_step_matches_dense_training_on_tiny_model() -> None:
    torch.manual_seed(0)

    model_cfg = ModelConfig(
        vocab_size=128,
        hidden_size=32,
        num_layers=4,
        num_heads=4,
        mlp_hidden_size=128,
        max_seq_len=16,
        tie_word_embeddings=False,
    )
    train_cfg = TrainingConfig(
        batch_size=2,
        seq_len=16,
        total_steps=1,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        grad_clip=10.0,
        checkpoint_group_size=2,
        compute_dtype="float32",
        master_dtype="float32",
        device="cpu",
        log_interval=1,
        seed=0,
        save_group_inputs_on_cpu=True,
        pin_group_inputs=False,
        use_double_buffer=False,
    )
    config = Config(model=model_cfg, data=DataConfig(), training=train_cfg)

    base_model = MiniTransformerLM(model_cfg)
    streaming_model = copy.deepcopy(base_model)

    batch = {
        "input_ids": torch.randint(0, model_cfg.vocab_size, (train_cfg.batch_size, train_cfg.seq_len), dtype=torch.long),
        "labels": None,
    }
    batch["labels"] = batch["input_ids"].clone()

    dense_loss, dense_state = _dense_step(base_model, batch, lr=train_cfg.learning_rate)

    trainer = StreamingTrainer(config, model=streaming_model)
    streaming_metrics = trainer.train_step(batch)
    streaming_state = {name: tensor.detach().clone() for name, tensor in trainer.model.state_dict().items()}

    assert abs(streaming_metrics.loss - dense_loss) < 1e-5
    for name, dense_tensor in dense_state.items():
        assert torch.allclose(streaming_state[name], dense_tensor, atol=1e-5), name
