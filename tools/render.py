"""Render chat examples through the exact Qwen non-thinking template.

Single source of truth for tokenization and label masking; the trainer,
preflight, eval, and tests all consume this module. The full text is
composed as prefix + assistant content + suffix so the label boundary is
deterministic and inspectable; assertions fail closed on template drift.
"""
import hashlib


class UnsupportedModelFamily(Exception):
    pass


MODEL_FAMILIES = {
    'qwen3.5': {
        'kwargs': {'enable_thinking': False},
        'suffix': '<|im_end|>\n',
        'nonthinking_tail': '<think>\n\n</think>\n\n',
    },
}


def family_for(model_name):
    name = model_name.lower()
    for family in MODEL_FAMILIES:
        if family in name:
            return family
    raise UnsupportedModelFamily(
        f'no non-thinking template mapping for {model_name}; '
        f'known families: {sorted(MODEL_FAMILIES)}')


def template_hash(tokenizer):
    return 'sha256:' + hashlib.sha256(
        tokenizer.chat_template.encode('utf-8')).hexdigest()


def render_texts(tokenizer, messages, model_name):
    fam = MODEL_FAMILIES[family_for(model_name)]
    if messages[-1]['role'] != 'assistant':
        raise ValueError('last message must be the assistant completion')
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True,
        **fam['kwargs'])
    if not prefix.endswith(fam['nonthinking_tail']):
        raise ValueError(
            'generation prefix does not end with the non-thinking tail; '
            'template drift, refusing to guess label boundaries')
    return prefix, prefix + messages[-1]['content'] + fam['suffix']


def encode_example(tokenizer, messages, model_name, max_length):
    prefix, full = render_texts(tokenizer, messages, model_name)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
    full_ids = tokenizer(full, add_special_tokens=False)['input_ids']
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError('prefix tokenization is not a prefix of the full '
                         'tokenization; label mask would be misaligned')
    if len(full_ids) > max_length:
        raise ValueError(f'example is {len(full_ids)} tokens, over {max_length}')
    if len(full_ids) == len(prefix_ids):
        raise ValueError('no assistant tokens carry loss')
    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
    return {'input_ids': full_ids, 'labels': labels, 'length': len(full_ids)}
