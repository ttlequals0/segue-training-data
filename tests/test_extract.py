from extract import classify_window, partition_markers, resolve_corrections


def marker(start, end, **kw):
    return {'start': start, 'end': end, 'category': 'sponsor', **kw}


def correction(type_, original=None, corrected=None, human=True):
    return {'type': type_, 'original': original, 'corrected': corrected,
            'human': human}


def bounds(start, end):
    return {'start': start, 'end': end}


def test_partition_markers_blocks_pending_and_uncategorized():
    markers = [
        marker(0, 30),
        marker(40, 60, held_for_review=True, was_cut=False),
        marker(70, 90, category=None),
        marker(100, 120, was_cut=False),
    ]
    cut, blocking = partition_markers(markers)
    assert [m['start'] for m in cut] == [0]
    assert [m['start'] for m in blocking] == [40, 70]


def test_partition_markers_rejects_missing_bounds():
    assert partition_markers([{'category': 'sponsor'}]) is None


def test_resolve_matches_within_tolerance_and_counts_stale():
    markers = [marker(10.0, 40.0), marker(100.0, 130.0, was_cut=False)]
    resolved, stale = resolve_corrections([
        correction('confirm', bounds(10.3, 39.8)),
        correction('false_positive', bounds(100, 130)),
        correction('false_positive', bounds(500, 530)),
    ], markers)
    assert stale == 1
    assert [r['label'] for r in resolved] == ['confirm', 'false_positive']
    assert not any(r['drop'] for r in resolved)


def test_resolve_labels_trims_and_auto_approvals():
    markers = [marker(10.0, 35.0)]
    resolved, _ = resolve_corrections([
        correction('confirm', bounds(10, 40), bounds(10, 35), human=False),
        correction('confirm', bounds(10, 35), bounds(10, 35)),
        correction('create', None, bounds(10, 35)),
    ], markers)
    assert [r['label'] for r in resolved] == [
        'auto_confirm_trimmed', 'confirm', 'create']


def test_resolve_flags_rejected_but_still_cut():
    resolved, _ = resolve_corrections(
        [correction('false_positive', bounds(0, 20))], [marker(0, 20)])
    assert resolved[0]['drop'] is True


def test_classify_window_precedence():
    assert classify_window([]) == ('machine_accepted', False, False)
    assert classify_window(['auto_confirm_trimmed']) == ('machine_accepted', False, False)
    assert classify_window(['auto_confirm', 'false_positive']) == ('hard_negative', True, True)
    assert classify_window(['boundary_adjustment', 'false_positive']) == ('human_verified', True, True)
    assert classify_window(['confirm']) == ('human_verified', True, False)
