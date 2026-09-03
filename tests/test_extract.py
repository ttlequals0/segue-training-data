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
    hits, stale = resolve_corrections([
        correction('confirm', bounds(10.3, 39.8)),
        correction('false_positive', bounds(100, 130)),
        correction('false_positive', bounds(500, 530)),
    ], markers)
    assert stale == 1
    assert [(h['label'], h['action']) for h in hits] == [
        ('confirm', 'keep'), ('false_positive', 'keep')]


def test_resolve_labels_trims_and_auto_approvals():
    markers = [marker(10.0, 35.0), marker(50.0, 60.0), marker(70.0, 80.0)]
    hits, _ = resolve_corrections([
        correction('confirm', bounds(10, 40), bounds(10, 35), human=False),
        correction('confirm', bounds(50, 60)),
        correction('create', None, bounds(70, 80)),
    ], markers)
    assert [h['label'] for h in hits] == [
        'auto_confirm_trimmed', 'confirm', 'create']


def test_resolve_drops_rejected_span_that_is_still_cut():
    hits, _ = resolve_corrections(
        [correction('false_positive', bounds(0, 20))], [marker(0, 20)])
    assert [h['action'] for h in hits] == ['drop']


def test_resolve_false_positive_by_coverage_of_the_marker():
    rejected = marker(714.6, 912.2, was_cut=False)
    redetected = marker(737.1, 912.2)
    superset = marker(700.0, 1300.0)
    elsewhere = marker(1400, 1500)
    hits, _ = resolve_corrections(
        [correction('false_positive', bounds(714.6, 912.2))],
        [rejected, redetected, superset, elsewhere])
    assert [h['action'] for h in hits] == ['keep', 'drop', 'block']
    assert hits[2]['marker'] is superset


def test_resolve_newest_decision_per_marker_wins():
    m = marker(0, 20)
    hits, _ = resolve_corrections([
        correction('confirm', bounds(0, 20)),
        correction('false_positive', bounds(0, 20)),
    ], [m])
    assert [(h['label'], h['action']) for h in hits] == [('false_positive', 'drop')]
    hits, _ = resolve_corrections([
        correction('boundary_adjustment', bounds(0, 20), bounds(0, 15)),
        correction('confirm', bounds(0, 20)),
    ], [m])
    assert [(h['label'], h['action']) for h in hits] == [('confirm', 'keep')]


def test_resolve_ignores_auto_approvals_that_do_not_keep():
    hits, stale = resolve_corrections([
        correction('confirm', bounds(0, 20), human=False),
        correction('confirm', bounds(100, 120), bounds(100, 110), human=False),
    ], [marker(0, 20, was_cut=False), marker(100, 120)])
    assert hits == []
    assert stale == 0


def test_resolve_prefers_cut_twin_and_blocks_uncut_positive():
    cut = marker(0, 20)
    twin = marker(0, 20, was_cut=False)
    hits, _ = resolve_corrections(
        [correction('confirm', bounds(0, 20))], [twin, cut])
    assert [(h['marker'] is cut, h['action']) for h in hits] == [(True, 'keep')]
    hits, _ = resolve_corrections(
        [correction('confirm', bounds(0, 20))], [twin])
    assert [h['action'] for h in hits] == ['block']


def test_resolve_blocks_unapplied_adjustment():
    hits, stale = resolve_corrections(
        [correction('boundary_adjustment', bounds(2309.6, 2415.6),
                    bounds(2309.6, 2400.0))],
        [marker(2309.6, 2415.6)])
    assert stale == 0
    assert [h['action'] for h in hits] == ['block']


def test_resolve_blocks_positive_that_covers_a_different_span():
    inside = marker(20, 30)
    hits, stale = resolve_corrections(
        [correction('create', None, bounds(10, 40))], [inside, marker(200, 300)])
    assert stale == 0
    assert [(h['marker'] is inside, h['action']) for h in hits] == [(True, 'block')]
    _, stale = resolve_corrections(
        [correction('confirm', bounds(10, 40))], [marker(10, 100)])
    assert stale == 1


def test_classify_window_precedence():
    assert classify_window([]) == ('machine_accepted', False, False)
    assert classify_window(['auto_confirm_trimmed']) == ('machine_accepted', False, False)
    assert classify_window(['auto_confirm', 'false_positive']) == ('hard_negative', True, True)
    assert classify_window(['boundary_adjustment', 'false_positive']) == ('human_verified', True, True)
    assert classify_window(['confirm']) == ('human_verified', True, False)
    assert classify_window(['confirm_trimmed']) == ('human_verified', True, True)
