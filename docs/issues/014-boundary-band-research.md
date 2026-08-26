# 014 — Research track: the smoothed boundary band

**Status:** open
**Phase:** R — timeboxed to **half a day**, hard stop
**Start day 2–3, in parallel with Phase 2 member training — not after the pipeline is "done".**

## Why this is worth a slot at all

It is the **highest-variance item on the board, and the only line of work with a ceiling above the
herd.** Ranks 17–100 sit within 2e-05 of each other, largely re-emitting one shared file. Rank 1
(Chris Deotte, 0.97184) is **~3 sigma clear** of rank 50 — real, and unpublished; his four public
notebooks are clean starters with no special tricks.

Everything else in the roadmap is porting known-good work toward a known plateau. This is the only
task that could beat it.

**Timing matters**: Phase 2 members train for 17 min to ~2.9 h each, mostly unattended. That is
the natural window for this. Left until the pipeline is finished, it gets cut.

## The opening

The source dataset is a two-rule lookup:

```
y = 1  if daily_screen > 8 or social_media > 4
y = 0  if daily_screen <= 6 and social_media <= 4
       coin flip otherwise        # 1,025 rows, 0.4556 positive, XGB AUC 0.510 ± 0.033
```

Bayes-optimal on the original: **0.9888**. The generator *smoothed* it — the same rules score only
**0.835** on competition data, while a model trained on the previously-random band reaches
**0.896**. The band is a third of all synthetic rows, and it is where all residual error lives.

**Do not hard-code the rules.** They cap at 0.835. The generator is the signal; the rules only say
*where to look*.

## Leads, in expected-value order

- [ ] **1. Model the smoothed boundary explicitly.** Signed distance-to-threshold features for
      `daily_screen > 8`, `social_media > 4`, `daily_screen <= 6`, plus a band indicator. Check
      whether a band-specialised model earns stack weight. Precedent: the public Optuna fusion
      already applies rank-preserving local re-sorts inside the 3–6 h and 6–7.8 h bands, which is a
      cruder version of the same idea.
- [ ] **2. Impute the two rule drivers transductively.** `daily_screen_time_hours` is missing on
      ~14% of rows and `social_media_hours` on ~19% — exactly the rows where the boundary cannot be
      evaluated. Impute-alongside is already known to be worth +0.0012 generally; the question is
      whether it is worth more *on these two columns specifically*.
- [ ] **3. Per-regime specialisation.** A model restricted to complete rows scores **0.9708** vs
      **0.9643** on all rows. Train one on complete rows only and enter it as a stack member with a
      missingness-regime interaction. Cheap to test.
- [ ] **4. The unanswered forum question.** `gender`, `stress_level` and `academic_work_impact`
      have ~0 *global* mutual information with the target, yet all remaining error sits in the
      5–8.7 h band. Nobody has published whether they carry *conditional* signal inside it.

## Rules of engagement

- Every lead gets a row in `docs/experiments.md` **whether it works or not** — a negative result
  here is worth as much as a positive one, because it closes off a lead for the remaining days.
- Judge against the **local** noise floor from issue 003, not the corpus constant.
- The timebox is a hard stop. The known-good pipeline has priority; this is the option, not the
  plan.

## Exit criterion

Four ledger rows, and a decision recorded: either a band feature/member that earns stack weight,
or a written note that the band is exhausted and the remaining days go entirely to Phase 2–4.
