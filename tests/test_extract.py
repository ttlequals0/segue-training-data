from extract import classify_window, correction_label, partition_markers, resolve_corrections


def marker(start, end, **kw):
    return {'start': start, 'end': end, 'category': 'sponsor', **kw}


def correction(type_, original=None, corrected=None, human=True):
    return {'type': type_, 'original': original, 'corrected': corrected,
            'human': human}


def decided(hits):
    return [(correction_label(h['source']), h['action']) for h in hits]


def placed(hits):
    return [(h['marker']['start'], *d) for h, d in zip(hits, decided(hits))]


def bounds(start, end):
    return {'start': start, 'end': end}


def test_partition_markers_buckets():
    markers = [
        marker(0, 30),
        marker(40, 60, held_for_review=True, was_cut=False),
        marker(70, 90, category=None),
        marker(100, 120, was_cut=False),
        marker(130, 140, was_cut=False, action_applied='keep'),
        marker(150, 160, was_cut=False, validation={'decision': 'REVIEW'}),
        marker(170, 180, was_cut=False, validation={'decision': 'REVIEW'},
               reviewer_verdict='reject'),
    ]
    cut, blocking, undecided = partition_markers(markers)
    assert [m['start'] for m in cut] == [0]
    assert [m['start'] for m in blocking] == [70]
    assert [m['start'] for m in undecided] == [40, 130, 150]


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
    assert decided(hits) == [
        ('confirm', 'keep'), ('false_positive', 'keep')]


def test_resolve_labels_trims_and_auto_approvals():
    markers = [marker(10.0, 35.0), marker(50.0, 60.0), marker(70.0, 80.0)]
    hits, _ = resolve_corrections([
        correction('confirm', bounds(10, 40), bounds(10, 35), human=False),
        correction('confirm', bounds(50, 60)),
        correction('create', None, bounds(70, 80)),
    ], markers)
    assert [correction_label(h['source']) for h in hits] == [
        'auto_confirm_trimmed', 'confirm', 'create']


def test_resolve_drops_rejected_span_that_is_still_cut():
    hits, _ = resolve_corrections(
        [correction('false_positive', bounds(0, 20))], [marker(0, 20)])
    assert [h['action'] for h in hits] == ['drop']


def test_resolve_false_positive_by_coverage_of_the_marker():
    rejected = marker(714.6, 912.2, was_cut=False)
    redetected = marker(737.1, 912.2)
    superset = marker(700.0, 1300.0)
    neighbour = marker(600.0, 720.0, held_for_review=True, was_cut=False)
    elsewhere = marker(1400, 1500)
    hits, _ = resolve_corrections(
        [correction('false_positive', bounds(714.6, 912.2))],
        [rejected, redetected, superset, neighbour, elsewhere])
    assert [h['action'] for h in hits] == ['keep', 'drop', 'block']
    assert hits[2]['marker'] is superset


def test_resolve_rejection_of_neighbour_keeps_confirmed_marker():
    confirmed = marker(100, 200)
    hits, _ = resolve_corrections([
        correction('confirm', bounds(100, 200)),
        correction('false_positive', bounds(190, 230)),
    ], [confirmed, marker(190, 230, was_cut=False)])
    assert decided(hits) == [
        ('confirm', 'keep'), ('false_positive', 'keep')]


def test_resolve_neighbour_only_rejection_blocks_and_is_not_stale():
    hits, stale = resolve_corrections(
        [correction('false_positive', bounds(15, 40))], [marker(0, 20)])
    assert decided(hits) == [('false_positive', 'block')]
    assert stale == 0


def test_resolve_withdrawn_rejection_does_not_block_neighbour():
    hits, stale = resolve_corrections([
        correction('false_positive', bounds(0, 20)),
        correction('confirm', bounds(0, 20)),
    ], [marker(0, 20), marker(15, 40)])
    assert decided(hits) == [('confirm', 'keep')]
    assert stale == 0


def test_resolve_rejection_blocks_neighbour_kept_only_by_auto_approval():
    m = marker(100, 200)
    hits, stale = resolve_corrections([
        correction('confirm', bounds(100, 200), human=False),
        correction('false_positive', bounds(190, 230)),
    ], [m])
    assert decided(hits) == [('false_positive', 'block')]
    assert stale == 0


def test_resolve_auto_approval_does_not_override_human_rejection():
    hits, stale = resolve_corrections([
        correction('false_positive', bounds(0, 20)),
        correction('confirm', bounds(0, 20), human=False),
    ], [marker(0, 20), marker(15, 40)])
    assert placed(hits) == [
        (0, 'false_positive', 'drop'), (15, 'false_positive', 'block')]
    assert stale == 0


def test_resolve_auto_approval_does_not_override_human_positive():
    hits, _ = resolve_corrections([
        correction('confirm', bounds(10, 40)),
        correction('confirm', bounds(20, 30), human=False),
    ], [marker(20, 30)])
    assert decided(hits) == [('confirm', 'block')]
    hits, _ = resolve_corrections([
        correction('confirm', bounds(0, 20)),
        correction('confirm', bounds(0, 20), human=False),
    ], [marker(0, 20)])
    assert decided(hits) == [('confirm', 'keep')]


def test_resolve_concurring_rejection_does_not_withdraw_the_first():
    hits, _ = resolve_corrections([
        correction('false_positive', bounds(100, 400)),
        correction('false_positive', bounds(250, 450)),
    ], [marker(300, 400), marker(90, 105)])
    assert placed(hits) == [
        (300, 'false_positive', 'drop'), (90, 'false_positive', 'block')]


def test_resolve_re_rejected_marker_revives_the_wider_rejection():
    hits, _ = resolve_corrections([
        correction('false_positive', bounds(100, 400)),
        correction('confirm', bounds(100, 200)),
        correction('confirm', bounds(300, 400)),
        correction('false_positive', bounds(300, 400)),
    ], [marker(100, 200), marker(300, 400), marker(90, 105)])
    assert placed(hits) == [
        (100, 'confirm', 'keep'), (300, 'false_positive', 'drop'),
        (90, 'false_positive', 'block')]


def test_resolve_partly_withdrawn_rejection_still_blocks():
    hits, _ = resolve_corrections([
        correction('false_positive', bounds(700, 1300)),
        correction('confirm', bounds(700, 900)),
    ], [marker(700, 900), marker(1000, 1300), marker(1290, 1400)])
    assert placed(hits) == [
        (700, 'confirm', 'keep'), (1000, 'false_positive', 'drop'),
        (1290, 'false_positive', 'block')]


def test_resolve_rejections_sharing_a_neighbour_are_not_stale():
    hits, stale = resolve_corrections([
        correction('false_positive', bounds(15, 40)),
        correction('false_positive', bounds(18, 45)),
    ], [marker(0, 20)])
    assert decided(hits) == [('false_positive', 'block')]
    assert stale == 0


def test_resolve_ignores_rejection_without_bounds():
    hits, stale = resolve_corrections(
        [correction('false_positive', None)], [marker(0, 20)])
    assert hits == [] and stale == 1


def test_resolve_newest_decision_per_marker_wins():
    m = marker(0, 20)
    hits, _ = resolve_corrections([
        correction('confirm', bounds(0, 20)),
        correction('false_positive', bounds(0, 20)),
    ], [m])
    assert decided(hits) == [('false_positive', 'drop')]
    hits, _ = resolve_corrections([
        correction('boundary_adjustment', bounds(0, 20), bounds(0, 15)),
        correction('confirm', bounds(0, 20)),
    ], [m])
    assert decided(hits) == [('confirm', 'keep')]


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
    assert [(h['marker'] is cut, h['action']) for h in hits] == [
        (False, 'keep'), (True, 'keep')]
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


def hit(type_, corrected=None, human=True):
    return {'source': correction(type_, bounds(0, 10), corrected, human)}


def test_classify_window_precedence():
    assert classify_window([]) == ('machine_accepted', False, False)
    assert classify_window([hit('confirm', bounds(0, 8), human=False)]) == ('machine_accepted', False, False)
    assert classify_window([hit('confirm', human=False), hit('false_positive')]) == ('hard_negative', True, True)
    assert classify_window([hit('boundary_adjustment'), hit('false_positive')]) == ('human_verified', True, True)
    assert classify_window([hit('confirm')]) == ('human_verified', True, False)
    assert classify_window([hit('confirm', bounds(0, 8))]) == ('human_verified', True, True)
