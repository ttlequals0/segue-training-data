import json
import os

import pytest

transformers = pytest.importorskip("transformers")

import render  # noqa: E402

TEST_MODEL = os.environ.get("SEGUE_TEST_TOKENIZER", "Qwen/Qwen3.5-4B")


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained(TEST_MODEL)
    except OSError as e:
        pytest.skip(f"tokenizer unavailable offline: {e}")


MESSAGES = [
    {"role": "system", "content": "You detect ads. Answer with a JSON array."},
    {"role": "user", "content": "[0.0s - 10.0s] buy stuff at example.com"},
    {"role": "assistant", "content": json.dumps(
        [{"start": 0.0, "end": 10.0, "confidence": 0.9, "category": "sponsor",
          "reason": "promo read", "end_text": "at example.com"}],
        separators=(",", ":"))},
]


def test_family_for():
    assert render.family_for("Qwen/Qwen3.5-9B") == "qwen3.5"
    with pytest.raises(render.UnsupportedModelFamily):
        render.family_for("Qwen/Qwen3.8-27B")
    with pytest.raises(render.UnsupportedModelFamily):
        render.family_for("meta-llama/Llama-3-8B")


@pytest.mark.tokenizer
def test_prefix_is_nonthinking(tokenizer):
    prefix, full = render.render_texts(tokenizer, MESSAGES, TEST_MODEL)
    assert prefix.endswith("<think>\n\n</think>\n\n")
    assert full.startswith(prefix)
    assert full.endswith("<|im_end|>\n")


@pytest.mark.tokenizer
def test_labels_mask_exactly_the_prefix(tokenizer):
    enc = render.encode_example(tokenizer, MESSAGES, TEST_MODEL, 16384)
    n_masked = sum(1 for t in enc["labels"] if t == -100)
    n_target = len(enc["labels"]) - n_masked
    assert n_target > 0
    target_ids = [t for t in enc["labels"] if t != -100]
    decoded = tokenizer.decode(target_ids)
    assert decoded == MESSAGES[2]["content"] + "<|im_end|>\n"
    assert enc["input_ids"][:n_masked] == tokenizer(
        render.render_texts(tokenizer, MESSAGES, TEST_MODEL)[0],
        add_special_tokens=False)["input_ids"]


@pytest.mark.tokenizer
def test_truncation_raises(tokenizer):
    with pytest.raises(ValueError, match="tokens"):
        render.encode_example(tokenizer, MESSAGES, TEST_MODEL, 16)
