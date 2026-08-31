# Phase 2 data plan

Phase 1 shipped a tier B model on 200 windows. This plan covers what to
extract next and why, measured against the source instance rather than
estimated. Inventory taken 2026-08-30 from a database copy of 474 processed
episodes with retained transcripts across 28 feeds.

## What phase 1 got wrong, and why

Recall was 0.592 against precision 0.738. The failures were not random:

| Bucket | Detection rate |
|---|---|
| Short ads (<30s) | 0.25 (n=8) |
| Medium (30 to 90s) | 0.47 (n=15) |
| Long (>=90s) | 0.68 (n=28) |
| Post-roll | 0.36 (n=11) |

The training data explains all of it. Only **3 of 200 windows contained more
than one ad**, so the model rarely saw two spans labeled separately in one
window and learned to merge adjacent spots into a single span. Every other ad
in a merged block then scores as a false negative. Boundary bias agrees: start
-7.5s, end +4.9s, extending outward past the real ad. Short ads had 10
training examples, and detection rate tracks example count almost linearly.

## What the instance actually holds

| Asset | Available | Used in phase 1 |
|---|---|---|
| Windows | ~5,700 | 200 |
| Feeds | 28 | 17 |
| Cut markers | 1,310 | 75 |
| Held-for-review markers | 260 | 0 |
| Rejected markers | 760 | 0 |
| Multi-ad windows | ~280 | 3 |
| Short ads (<30s) | 124 | 10 |
| Medium ads | 415 | 20 |
| Long ads | 771 | 45 |

Categories across all cut markers: 586 sponsor, 29 outro, 16 self_promo, 6
cross_promo, 4 intro, 3 recap, and 666 uncategorized. That uncategorized half
is why the phase 1 extractor discarded 141 episodes outright.

Cleaning up what already exists is worth roughly 28 times the training data,
and it lands where the model is weakest: 280 multi-ad windows against the 3
that caused the merging, and 124 short ads against 10.

## Work items

1. Keep held-for-review episodes. The phase 1 extractor drops an entire
   episode when any marker is pending review, which cost 156 episodes. Filter
   per marker instead: cut markers stay, pending ones are excluded from the
   completion, and the rest of the episode is still usable.
2. Backfill the 666 uncategorized markers. Rules first, matching sponsor
   names and community patterns, then LLM few-shot seeded from the markers
   that already carry a category. Write back through a new MinusPod endpoint,
   with `category_source` recorded per example so an ablation can drop them.
3. Add the 760 rejected markers as hard negatives. These are spans the
   pipeline or a person explicitly ruled not an ad, on real audio and in
   context. They are the highest-value negative signal available and none of
   it is in the dataset today. The window is kept with the rejected span
   absent from the completion.
4. Re-window for multi-ad coverage. Emit each episode at several window
   offsets so ad blocks split differently, turning single-ad windows into
   multi-ad ones and moving ads to window edges. This targets the merging
   behavior directly and costs no new episodes.
5. Prefix-invariant rendering. Render every example under both the
   thinking-open and thinking-closed assistant prefixes against the same JSON
   target, so no serving flag is load-bearing.
6. Restore `audio_context`. Phase 1 omitted the audio cue and volume block
   that production prepends, and flagged it in provenance. Rebuild it from the
   stored audio analysis so training matches inference.
7. Pattern injection for short ads. Splice community patterns into clean
   transcript stretches with correct labels by construction. Keeps synthetic
   examples under about a quarter of the mix and out of eval entirely.

## Coverage targets

| Bucket | Phase 1 | Target | Source |
|---|---|---|---|
| Total windows | 200 | 5,000+ | items 1 and 2 |
| Multi-ad windows | 3 | 500+ | items 3 and 4 |
| Short ads | 10 | 250+ | items 2 and 7 |
| Hard negatives | 0 | 700+ | item 3 |
| Feeds | 17 | 28 | item 1 |
| `interaction` | 0 | 100 | not available locally |
| `cross_promo` | 2 | 100 | thin locally, 6 total |

## The two gaps the instance cannot fill

`interaction` has zero examples and `cross_promo` has six, yet the production
prompt requires both categories. The backfill may surface some, since half of
all markers are uncategorized. Whatever remains is the concrete ask for
contributors, and it is worth waiting for the backfill before making that ask
so the request names real gaps.

## On growth and the back catalog

New episodes arrive at about 10.6 per day on average, bursty rather than
steady, which is roughly 125 windows and 30 ad spans a day. Useful
compounding, but small next to the 28x already sitting in the database.
Cleanup is the lever, not growth.

The feeds expose 15,248 episodes against 530 processed, so the back catalog is
effectively unlimited. At the instance's measured cost per episode, processing
it would run into thousands of dollars, so it is not worth reaching for until
the existing episodes are exhausted.

## Training changes to pair with the data

Hold LoRA rank while the data changes so the two effects stay separable, then
sweep rank 16, 32, and 64 on the finished dataset. If rank 64 matches rank 16,
capacity is not the constraint and a full fine-tune will not help either. The
prediction going in is that the sweep comes back flat, because train nll 0.275
against test nll 0.282 says the model already fits what it was shown, and
because span merging reads as a label-distribution problem rather than a
capacity one. A full fine-tune is worth its cost only above roughly 10,000
windows, and only if the sweep says capacity binds.
