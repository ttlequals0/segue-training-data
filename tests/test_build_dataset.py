import random

from build_dataset import split_examples


def ex(feed, tier="machine_accepted", n=0):
    return {"id": f"{feed}/ep/w{n}", "tier": tier,
            "source": {"feed": feed, "episode_id": "ep", "window": n},
            "prompt": {"system": "sha256:" + "0" * 64, "user": "u",
                       "window_start": 0.0, "window_end": 1.0},
            "completion": []}


def test_val_feeds_route_to_val():
    train, val, down = split_examples(
        [ex("a"), ex("b")], {"b"}, {}, random.Random(13))
    assert [e["id"] for e in train] == ["a/ep/w0"]
    assert [e["id"] for e in val] == ["b/ep/w0"]
    assert down == 0


def test_tier_weights_never_touch_val():
    examples = [ex("a", n=i) for i in range(50)] + [ex("b", n=i) for i in range(50)]
    train, val, down = split_examples(
        examples, {"b"}, {"machine_accepted": 0.0}, random.Random(13))
    assert train == [] and down == 50
    assert len(val) == 50


def test_split_is_disjoint_and_complete():
    examples = [ex("a", n=i) for i in range(10)] + [ex("b", n=i) for i in range(10)]
    train, val, down = split_examples(examples, {"b"}, {}, random.Random(13))
    ids = {e["id"] for e in train} | {e["id"] for e in val}
    assert len(ids) == 20 and down == 0
