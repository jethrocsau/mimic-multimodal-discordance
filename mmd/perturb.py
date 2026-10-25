"""What to perturb: the perturbation strategies, and the configuration list they generate for a modality set.

Every strategy replaces a SUBSET of the set's modalities and keeps the rest at the stay's own baseline record.
Subsets run from one modality up to `max_order` modalities at once (combinations, not permutations: {x, n} is
one input whichever order you name it), so a set of k modalities gets sum_{r=1..max_order} C(k, r) subsets.

    strategy              family (tables)   the perturbed modalities become
    ablate                ablate            none (removed from the input)
    within_patient        temporal          the stay's own latest follow-up record
    matched_patient       matched           a same-outcome donor's baseline record
    matched_label_swap    counterfactual    an opposite-outcome donor's baseline record
    cohort_mean           mean              signals: the cohort-mean encoder output; note: none

Rules that keep every configuration informative:
  - donor, mean and ablate subsets stop at k - 1 modalities: replacing every modality gives the donor's own
    reference (or an input with no patient in it); within-patient may replace all k (the stay's later self);
  - a joint donor swap takes every perturbed modality from the SAME donor (one donor plan per recipient);
  - the core arms always run, whatever strategies are chosen: the reference and every "only modality m" input
    (ablate everything else). The zoo's information share needs them.

A configuration row: level_id, config (`<family>_<slot>`, e.g. counterfactual_x+n), family, slot (perturbed
modalities joined with '+'), order (how many), core, factor (prompt factor or ''), stay_id, draw, donor, record.
"""
import itertools
from dataclasses import dataclass
from datetime import datetime, timezone
from math import comb
from typing import Dict, List, Sequence

import pandas as pd

from .cohort import CONFIG_COLUMNS, DonorPlan, Manifest, TierCohort
from .modsets import NOTE, ModalitySet, slot_name
from .records import donor_slot, parse_record, record_id


@dataclass(frozen=True)
class Strategy:
    name: str                     # what you type on the command line
    family: str                   # what the tables call it
    donor: bool                   # draws records from the donor plan
    whole_set: bool               # may perturb every modality of the set at once
    becomes: str                  # what the perturbed modalities are replaced with

    def max_size(self, k: int, max_order: int) -> int:
        return min(max_order, k if self.whole_set else k - 1)


STRATEGIES: Dict[str, Strategy] = {s.name: s for s in [
    Strategy('ablate', 'ablate', False, False, 'removed'),
    Strategy('within_patient', 'temporal', False, True, "the stay's own latest follow-up record"),
    Strategy('matched_patient', 'matched', True, False, "a same-outcome donor's baseline record"),
    Strategy('cohort_mean', 'mean', False, False, 'the cohort-mean encoder output (note: removed)'),
    Strategy('matched_label_swap', 'counterfactual', True, False, "an opposite-outcome donor's baseline record"),
]}
BY_FAMILY = {s.family: s for s in STRATEGIES.values()}
FAMILIES = [s.family for s in STRATEGIES.values()]             # table order
DONOR_FAMILIES = [s.family for s in STRATEGIES.values() if s.donor]
REFERENCE = 'reference'


def resolve(strategies='all') -> List[Strategy]:
    """'all' | strategy names | family names (a list, or one comma-separated string) -> strategies, in order."""
    if isinstance(strategies, str):
        strategies = strategies.split(',')
    names = {n.strip() for n in strategies if n and n.strip()}
    if not names or 'all' in names:
        return list(STRATEGIES.values())
    lookup = {k: s for s in STRATEGIES.values() for k in (s.name, s.family)}
    bad = sorted(names - set(lookup))
    if bad:
        raise ValueError(f'unknown strategy {bad}; choose from {list(STRATEGIES)} or all')
    chosen = {lookup[n].name for n in names}
    return [s for s in STRATEGIES.values() if s.name in chosen]


def check_order(max_order: int, k: int) -> int:
    max_order = int(max_order)
    if not 1 <= max_order <= k:
        raise ValueError(f'max_order must be between 1 and the number of modalities ({k}), got {max_order}')
    return max_order


def subsets(mods: Sequence[str], hi: int, lo: int = 1):
    """Every subset of `mods` with lo..hi members (combinations), smallest first."""
    return [c for r in range(lo, hi + 1) for c in itertools.combinations(mods, r)]


def count_configs(k: int, strategies='all', max_order: int = 2, n_draws: int = 5) -> pd.DataFrame:
    """Configurations per recipient for each family (an upper bound: temporal needs follow-up records)."""
    max_order = check_order(max_order, k)
    chosen = {s.family for s in resolve(strategies)}
    rows = [dict(family=REFERENCE, subsets=1, draws=1)]
    for s in STRATEGIES.values():
        hi = s.max_size(k, max_order)
        n = sum(comb(k, r) for r in range(1, hi + 1)) if s.family in chosen else 0
        if s.family == 'ablate' and k - 1 > (hi if s.family in chosen else 0):
            n += k                                            # the core "only m" arms
        if n:
            rows.append(dict(family=s.family, subsets=n, draws=n_draws if s.donor else 1))
    out = pd.DataFrame(rows)
    out['configs'] = out.subsets * out.draws
    return out


# ── the configuration list ───────────────────────────────────────────────────
def build_configs(cohort: TierCohort, plan: DonorPlan, strategies='all', max_order: int = 2,
                  draws: int = None) -> pd.DataFrame:
    """Every configuration one tier needs: one row per (config, recipient, draw)."""
    t, mods = cohort.table, tuple(cohort.mods)
    k = len(mods)
    max_order = check_order(max_order, k)
    chosen = {s.family: s for s in resolve(strategies)}
    base = {m: 'first' for m in mods}
    rows = []

    def add(family, sub, stay, slots, draw=0, donor=None):
        slot = slot_name(sub)
        config = 'ref' if family == REFERENCE else f'{family}_{slot}'
        core = family == REFERENCE or (family == 'ablate' and len(sub) == k - 1)
        rows.append((cohort.level, config, family, slot, len(sub), core, '', int(stay), draw, donor,
                     record_id(stay, slots)))

    ablate_sets = list(dict.fromkeys(
        (subsets(mods, STRATEGIES['ablate'].max_size(k, max_order)) if 'ablate' in chosen else [])
        + list(itertools.combinations(mods, k - 1))))
    own = [(fam, subsets(mods, chosen[fam].max_size(k, max_order))) for fam in ('temporal', 'mean') if fam in chosen]
    for s in t.itertuples():
        add(REFERENCE, (), s.stay_id, base)
        for sub in ablate_sets:
            add('ablate', sub, s.stay_id, {**base, **{m: 'none' for m in sub}})
        for fam, subs in own:
            for sub in subs:
                if fam == 'temporal':
                    if all(getattr(s, f'{m}_has_last') for m in sub):
                        add(fam, sub, s.stay_id, {**base, **{m: 'last' for m in sub}})
                else:                                         # mean: a signal gets the cohort mean, the note goes
                    add(fam, sub, s.stay_id, {**base, **{m: 'none' if m == NOTE else 'mean' for m in sub}})
    p = plan.plan[plan.plan.draw < (draws or plan.n_draws)]
    for fam in DONOR_FAMILIES:
        if fam not in chosen:
            continue
        subs = subsets(mods, chosen[fam].max_size(k, max_order))
        for r in p[p.family == fam].itertuples():
            for sub in subs:                                  # every perturbed modality from the SAME donor
                add(fam, sub, r.recipient, {**base, **{m: donor_slot(r.donor) for m in sub}},
                    draw=r.draw, donor=r.donor)
    return pd.DataFrame(rows, columns=CONFIG_COLUMNS)


def factor_configs(configs: pd.DataFrame, levels=('L1',), orders=('flip',), paras=(1, 2)) -> pd.DataFrame:
    """Prompt-factor copies (slot order 'flip' = the other order from the model's default, and paraphrases) of
    the reference and the single-modality counterfactual swaps of some tiers, so the factors add ~30% rather
    than 6x. The copies carry `~order` / `~para` in the record id and `order=flip` / `para=1` in `factor`."""
    base = configs[configs.level_id.isin(levels) & configs.factor.eq('')
                   & ((configs.family == REFERENCE) | ((configs.family == 'counterfactual') & (configs.order == 1)))]
    out = []
    for kind, values in (('order', orders), ('para', paras)):
        for v in values:
            c = base.copy()
            c['factor'] = f'{kind}={v}'
            c['record'] = [record_id(*parse_record(r), options={kind: str(v)}) for r in c.record]
            out.append(c)
    return pd.concat(out, ignore_index=True) if out else base.iloc[:0]


def build_manifest(cfg, modset: ModalitySet, ad: pd.DataFrame, slots: Dict[str, pd.DataFrame],
                   donor_plan_dir: str = None, strategies=None, max_order: int = None, factors: bool = True,
                   verbose=True) -> Manifest:
    """Tiers, folds, donors (reused from saved plans when they match) and every configuration, for one set."""
    strategies = [s.name for s in resolve(strategies or cfg.strategies)]
    max_order = check_order(max_order or cfg.max_order, modset.k)
    missing = set(modset.mods) - set(slots)
    if missing:
        raise ValueError(f'{modset.name}: no slot table for {sorted(missing)}')
    slots = {m: slots[m] for m in modset.mods}
    cohorts = {lv: TierCohort(ad, lv, slots, n_splits=cfg.n_splits) for lv in cfg.levels}
    plan_path = lambda lv: f'{donor_plan_dir}/donor_plan_{modset.name}_{lv}.csv' if donor_plan_dir else None
    plans = {lv: DonorPlan(c, cfg.n_draws, path=plan_path(lv), verbose=verbose) for lv, c in cohorts.items()}
    configs = pd.concat([build_configs(c, plans[lv], strategies, max_order, draws=cfg.n_draws)
                         for lv, c in cohorts.items()], ignore_index=True)
    if factors:
        configs = pd.concat([configs, factor_configs(configs[configs.draw < cfg.n_draws_model],
                                                     levels=cfg.levels[:1])], ignore_index=True)
    meta = dict(modset=modset.name, mods=list(modset.mods), cohort_version=cfg.cohort_version,
                levels=list(cfg.levels), strategies=strategies, max_order=max_order, n_splits=cfg.n_splits,
                n_draws=cfg.n_draws, n_draws_model=cfg.n_draws_model,
                donor_plans_from={lv: plan_path(lv) for lv in cohorts} if donor_plan_dir else {},
                created=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                hash=Manifest.digest(configs, cohorts))
    return Manifest(modset, cohorts, plans, configs, meta)
