"""Span policy: one span is one contiguous ad break (sub-15s gaps merge)."""

GAP_SECONDS = 15.0


def merge_gaps(spans, gap=GAP_SECONDS):
    """Merge dict spans whose gap is under `gap` seconds."""
    if not spans:
        return []
    ordered = sorted((dict(s) for s in spans),
                     key=lambda s: (s["start"], s["end"]))
    out = [ordered[0]]
    for s in ordered[1:]:
        cur = out[-1]
        if s["start"] - cur["end"] < gap:
            if s["end"] > cur["end"]:
                cur["end"] = s["end"]
                if "end_text" in s:
                    cur["end_text"] = s["end_text"]
            cur["confidence"] = max(cur["confidence"], s["confidence"])
        else:
            out.append(s)
    return out


def iou(a, b):
    overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    if overlap == 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - overlap
    return overlap / union if union > 0 else 0.0


def match_spans(predictions, truths, threshold=0.5):
    """Greedy one-to-one IoU matching; mirrors the MinusPod benchmark scorer."""
    pairs = []
    for pi, p in enumerate(predictions):
        for ti, t in enumerate(truths):
            score = iou(p, t)
            if score >= threshold:
                pairs.append((score, pi, ti))
    pairs.sort(key=lambda x: x[0], reverse=True)
    used_p, used_t, matches = set(), set(), []
    for score, pi, ti in pairs:
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        matches.append((pi, ti, score))
    return matches
