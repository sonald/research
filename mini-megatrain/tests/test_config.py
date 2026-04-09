from mini_megatrain.config import ModelConfig, estimate_parameter_count


def test_demo_1b_parameter_count_is_about_one_billion() -> None:
    config = ModelConfig(
        vocab_size=32768,
        hidden_size=1536,
        num_layers=24,
        num_heads=12,
        mlp_hidden_size=6144,
        max_seq_len=1024,
        tie_word_embeddings=False,
    )
    params = estimate_parameter_count(config)
    assert 990_000_000 <= params <= 1_020_000_000
