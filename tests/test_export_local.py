import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from export_local import checksum_dir, generations_match, compute_logit_max_diff  # noqa: E402


def tiny_pair():
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, vocab_size=128)
    peft_model = get_peft_model(LlamaForCausalLM(cfg), LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
        target_modules="all-linear"))
    for _, p in peft_model.named_parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=0.02)
    import copy
    merged = copy.deepcopy(peft_model).merge_and_unload()
    return peft_model.eval(), merged.eval()


def test_merged_matches_adapter_generations():
    a, b = tiny_pair()
    ids = torch.randint(0, 128, (1, 8))
    assert generations_match(a, b, ids, max_new_tokens=16)


def test_generations_match_detects_difference():
    a, _ = tiny_pair()
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(99)
    other = LlamaForCausalLM(LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=128)).eval()
    ids = torch.randint(0, 128, (1, 8))
    assert not generations_match(a, other, ids, max_new_tokens=16)


def test_checksum_dir(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y")
    sums = checksum_dir(tmp_path)
    assert set(sums) == {"a.bin", "sub/b.bin"}
    assert all(v.startswith("sha256:") for v in sums.values())


def test_logit_max_diff_adapter_vs_merged():
    a, b = tiny_pair()
    ids = [torch.randint(0, 128, (1, 8)), torch.randint(0, 128, (1, 8))]
    diff = compute_logit_max_diff(a, b, ids)
    assert diff >= 0.0
    assert diff < 1e-5


def test_logit_max_diff_detects_difference():
    a, _ = tiny_pair()
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(99)
    other = LlamaForCausalLM(LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=128)).eval()
    ids = [torch.randint(0, 128, (1, 8))]
    diff = compute_logit_max_diff(a, other, ids)
    assert diff > 0.1
