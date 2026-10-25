"""The perturbation charts: reference shifts, AUROC, within-patient swaps, and each modality's reliance share.

Every function takes the table it draws, returns the figure, and saves a PNG to ctx.fig_dir when given a name.
Colours and markers follow the slot (which modalities were perturbed), the same in every chart of one context.
"""
import re

import matplotlib.pyplot as plt
import numpy as np

from .. import fsio
from ..modsets import MOD_NAMES
from .effects import EFFECT_FAMILIES, GROUPS, LABEL_SPECIFIC, MEASURES, PALETTE, AnalysisContext, binned
from .probe import RAW
from ..perturb import FAMILIES


def ftag(s):
    return re.sub(r'[^0-9A-Za-z]+', '_', s).strip('_')


def clf_tag(clf):
    return 'raw' if clf == RAW else clf.split()[0].lower()


def _finish(ctx, fig, name, show):
    if name and ctx.fig_dir:
        print('saved', fsio.savefig(fig, f'{ctx.fig_dir}/{name}.png'))
    plt.show() if show else plt.close(fig)
    return fig


def _how(clf):
    return "the model's own P(Yes), no probe" if clf == RAW else 'probe fitted on the reference only'


def _levels(t):
    return [l for l in ['L1', 'L2', 'L3'] if l in set(t.level_id)]


def reference_chart(ctx: AnalysisContext, rs, readout, clf, name=None, show=False):
    """Arrows from each outcome group's reference risk to its risk after each perturbation."""
    t = rs[(rs.readout == readout) & (rs.clf == clf)]
    if t.empty:
        return None
    levels = _levels(t)
    rows = [(f, g) for f in EFFECT_FAMILIES for g, _ in GROUPS
            if not t[(t.family == f) & (t.group == g)].empty][::-1]
    slots = [s for s in ctx.slots if s in set(t.slot)]
    step = 0.8 / max(len(slots), 1)
    off = {s: (i - (len(slots) - 1) / 2) * step for i, s in enumerate(slots)}
    vals = np.r_[t.ref, t.swapped - t.delta_ci, t.swapped + t.delta_ci]
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    pad = 0.06 * max(hi - lo, 1e-6)
    fig, axes = plt.subplots(1, len(levels), figsize=(5.8 * len(levels), 0.3 * len(rows) * max(len(slots), 2) + 1.8),
                             sharey=True, squeeze=False)
    for ax, lv in zip(axes[0], levels):
        d = t[t.level_id == lv]
        for yi, (f, g) in enumerate(rows):
            if yi and rows[yi - 1][0] != f:
                ax.axhline(yi - 0.5, color='#d9d8d3', lw=1, zorder=0)
            for s in slots:
                r = d[(d.family == f) & (d.group == g) & (d.slot == s)]
                if r.empty:
                    continue
                r = r.iloc[0]
                col = ctx.style(s)[1]
                y = yi + off[s]
                ax.plot([r.ref], [y], marker='|', ms=9, mew=2, color='#7f7f7f', zorder=2)
                ax.annotate('', xy=(r.swapped, y), xytext=(r.ref, y), zorder=3,
                            arrowprops=dict(arrowstyle='-|>', color=col, lw=1.6, shrinkA=0, shrinkB=0,
                                            mutation_scale=10))
                ax.errorbar(r.swapped, y, xerr=r.delta_ci, fmt='none', ecolor=col, elinewidth=1,
                            capsize=2.5, alpha=0.8, zorder=3)
                ax.text(1.02, y, f'Δ {r.delta:+.3f} ± {r.delta_ci:.3f}  n={r.n}',
                        transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=6.5, color=col)
        ax.set_xlim(lo - pad, hi + pad)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_title(lv, loc='left'); ax.grid(False)
        ax.set_xlabel('mean P(Yes)' if clf == RAW else 'mean predicted 1-year risk')
    lab = {LABEL_SPECIFIC: 'label-specific (from matched)'}
    axes[0][0].set_yticks(range(len(rows)), [f'{lab.get(f, f)} · {g}' for f, g in rows])
    axes[0][0].set_ylim(-0.6, len(rows) - 0.4)
    handles = [plt.Line2D([], [], color=ctx.style(s)[1], lw=2, marker='>', ms=6, label=ctx.style(s)[0]) for s in slots]
    handles.append(plt.Line2D([], [], color='#7f7f7f', marker='|', ls='', ms=10, mew=2,
                              label='reference: own baseline records (arrow head = after the perturbation)'))
    fig.legend(handles=handles, ncol=min(len(handles), 4), loc='lower left', bbox_to_anchor=(0.01, 0.0),
               frameon=False, fontsize=8.5)
    fig.suptitle(f'{ctx.model_name} · {ctx.modset.label} · {readout} · {clf}: how far each perturbation moves '
                 f'predicted risk from the reference\npaired Δ per patient, {_how(clf)} · '
                 'counterfactual should push decedents left and survivors right', x=0.01, ha='left', fontsize=10.5)
    h = fig.get_figheight()
    fig.tight_layout(rect=[0, (0.35 + 0.22 * (len(handles) // 4 + 1)) / h, 1, 1 - 0.75 / h])
    return _finish(ctx, fig, name, show)


def auroc_chart(ctx: AnalysisContext, au, readout, clf, name=None, show=False):
    """AUROC at the reference and after each perturbation, one line per slot, CIs only."""
    t = au[(au.readout == readout) & (au.clf == clf)]
    if t.empty:
        return None
    levels = _levels(t)
    rows = [(f, o) for f in FAMILIES for o in sorted(t.order[t.family == f].unique())][::-1]
    slots = [s for s in ctx.slots if s in set(t.slot)]
    step = 0.8 / max(len(slots), 1)
    off = {s: ((len(slots) - 1) / 2 - i) * step for i, s in enumerate(slots)}     # first slot on top
    lo = min(0.45, np.nanmin(np.r_[t.auroc_ref, t.swap_lo]))
    hi = max(0.55, np.nanmax(np.r_[t.auroc_ref, t.swap_hi]))
    pad = 0.04 * (hi - lo)
    fig, axes = plt.subplots(1, len(levels), figsize=(6.6 * len(levels), 0.3 * len(rows) * max(len(slots), 3) + 1.6),
                             sharey=True, squeeze=False)
    for ax, lv in zip(axes[0], levels):
        d = t[t.level_id == lv]
        ax.axvline(0.5, color='#bdbdbd', ls='--', lw=1, zorder=0)
        for y, (f, o) in enumerate(rows):
            r = d[(d.family == f) & (d.order == o)]
            if r.empty:
                continue
            for q in r.itertuples():
                lab, col, mk = ctx.style(q.slot)
                yy = y + off[q.slot]
                ax.plot([q.auroc_ref], [yy], marker='|', ms=9, mew=2, color='#7f7f7f', zorder=2)
                ax.errorbar(q.auroc_swap, yy, xerr=[[q.auroc_swap - q.swap_lo], [q.swap_hi - q.auroc_swap]],
                            fmt=mk, color=col, ms=5, mec='white', elinewidth=1.3, capsize=3, zorder=4)
                ax.text(1.02, yy, f'Δ {q.d_auroc:+.3f} [{q.d_lo:+.3f}, {q.d_hi:+.3f}]',
                        transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=6.5, color=col)
            ax.text(0.01, y + 0.47, f'n={r.n.iloc[0]} ({r.died.iloc[0]} died)', transform=ax.get_yaxis_transform(),
                    fontsize=7, color='#52514e', va='top')
            if y:
                ax.axhline(y - 0.5, color='#d9d8d3', lw=1, zorder=0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(lv, loc='left'); ax.grid(False)
        ax.set_xlabel('AUROC (1-year mortality)')
    axes[0][0].set_yticks(range(len(rows)), [f'{f} · {o} modalit{"y" if o == 1 else "ies"}' for f, o in rows])
    axes[0][0].set_ylim(-0.5, len(rows) - 0.5)
    handles = [plt.Line2D([], [], color=ctx.style(s)[1], lw=1.4, marker=ctx.style(s)[2], ms=6,
                          label=f'{ctx.style(s)[0]}: AUROC and 95% CI') for s in slots]
    handles.append(plt.Line2D([], [], color='#7f7f7f', marker='|', ls='', ms=12, mew=2.5,
                              label='reference AUROC: own baseline records'))
    fig.legend(handles=handles, ncol=min(len(handles), 3), loc='lower left', bbox_to_anchor=(0.01, 0.0),
               frameon=False, fontsize=8.5)
    fig.suptitle(f'{ctx.model_name} · {ctx.modset.label} · {readout} · {clf}: AUROC at the reference and after each '
                 f'perturbation\nsame patients and reference within a row, {_how(clf)} · '
                 'donor rows: compare the slots with each other, not the distance to the reference',
                 x=0.01, ha='left', fontsize=10.5)
    h = fig.get_figheight()
    fig.tight_layout(rect=[0, (0.3 + 0.22 * (len(handles) // 3 + 1)) / h, 1, 1])
    return _finish(ctx, fig, name, show)


def temporal_chart(ctx: AnalysisContext, tp, readout, clf='Logistic regression', show=False):
    """Within-patient swap per modality: outcome pull against how much the record changed."""
    t = tp[tp.readout == readout]
    if t.empty:
        return None
    levels = _levels(t)
    fig, axes = plt.subplots(1, len(levels), figsize=(5.4 * len(levels), 4.2), squeeze=False, sharey=True)
    for j, lv in enumerate(levels):
        d = t[t.level_id == lv]
        ax = axes[0][j]
        ax.axhline(0, color='grey', ls='--', lw=1, zorder=1)
        for m in ctx.mods:
            lab, col, mk = ctx.style(m)
            b = binned(d.sim[d['mod'] == m], d.pull[d['mod'] == m])
            ax.errorbar(b.x, b.m, yerr=b.ci, fmt=mk + '-', color=col, ms=6, lw=2, capsize=3,
                        markeredgecolor='white', label=f'{MOD_NAMES[m]} (follow-up record)', zorder=3)
        n = d.stay_id.nunique()
        ax.set_title(f'{lv} · {n} paired patients, {int(d.drop_duplicates("stay_id").y.sum())} died', loc='left')
        ax.set_xlabel('cosine similarity, baseline vs follow-up (1 = unchanged)')
        ax.invert_xaxis(); ax.grid(False)
    axes[0][0].set_ylabel('outcome pull\n(Δ risk toward the true outcome)')
    axes[0][0].legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1.1), ncol=min(len(ctx.mods), 3), fontsize=9)
    fig.suptitle(f'{ctx.model_name} · {readout} · {clf}: within-patient swap of each modality on the same patients\n'
                 'does the follow-up record move risk the right way, by how much it changed the readout',
                 x=0.01, ha='left', fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.8])
    return _finish(ctx, fig, f'temporal_{ftag(readout)}_{clf_tag(clf)}', show)


def reliance_chart(ctx: AnalysisContext, rel, readout, clf, show=False):
    """Each modality's share of the total effect, per strategy and measure, with paired bootstrap CIs."""
    t = rel[(rel.readout == readout) & (rel.clf == clf) & rel.share.notna()]
    if t.empty:
        return None
    levels = _levels(t)
    rows = [(f, m) for m in MEASURES for f in EFFECT_FAMILIES if not t[(t.family == f) & (t.measure == m)].empty][::-1]
    mods = list(ctx.mods)
    step = 0.8 / max(len(mods), 1)
    off = {m: ((len(mods) - 1) / 2 - i) * step for i, m in enumerate(mods)}
    fig, axes = plt.subplots(1, len(levels), figsize=(5.6 * len(levels), 0.3 * len(rows) * max(len(mods), 2.5) + 2.0),
                             sharey=True, squeeze=False)
    for ax, lv in zip(axes[0], levels):
        d = t[t.level_id == lv]
        ax.axvline(1 / len(mods), color='grey', ls='--', lw=1, zorder=1)
        for yi, (f, me) in enumerate(rows):
            if yi and rows[yi - 1][1] != me:
                ax.axhline(yi - 0.5, color='#d9d8d3', lw=1, zorder=0)
            for i, m in enumerate(mods):
                r = d[(d.family == f) & (d.measure == me) & (d['mod'] == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                col, mk = PALETTE[i % len(PALETTE)]
                y = yi + off[m]
                ax.errorbar(r.share, y, xerr=[[r.share - r.share_lo], [r.share_hi - r.share]],
                            fmt=mk, color=col, ms=6, capsize=2, lw=1.2, markeredgecolor='white', zorder=3)
                ax.text(1.02, y, f'{r.share:.2f}', transform=ax.get_yaxis_transform(), ha='left',
                        va='center', fontsize=6.5, color=col)
        ax.set_xlim(-0.02, 1.02)
        ax.set_title(lv, loc='left'); ax.grid(False)
        ax.set_xlabel(f'share of the total effect (dashed: equal shares, 1/{len(mods)})')
    axes[0][0].set_yticks(range(len(rows)), [f'{f} · {m}' for f, m in rows])
    axes[0][0].set_ylim(-0.6, len(rows) - 0.4)
    handles = [plt.Line2D([], [], marker=PALETTE[i % len(PALETTE)][1], ls='', color=PALETTE[i % len(PALETTE)][0],
                          ms=7, label=MOD_NAMES[m]) for i, m in enumerate(mods)]
    fig.legend(handles=handles, ncol=len(handles), loc='lower left', bbox_to_anchor=(0.01, 0.0), frameon=False,
               fontsize=9)
    fig.suptitle(f'{ctx.model_name} · {ctx.modset.label} · {readout} · {clf}: which modality the model relies on\n'
                 'each modality perturbed alone, same patients and donors; share of the summed mean effect, '
                 'paired bootstrap 95% CI', x=0.01, ha='left', fontsize=10.5)
    h = fig.get_figheight()
    fig.tight_layout(rect=[0, 0.55 / h, 1, 1 - 0.75 / h])
    return _finish(ctx, fig, f'reliance_{ftag(readout)}_{clf_tag(clf)}', show)
