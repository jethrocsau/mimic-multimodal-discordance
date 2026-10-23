"""The swap charts (ported from the notebooks): reference shifts, AUROC, temporal pairs, paired dominance.

Every function takes the table it draws, returns the figure, and saves a PNG to ctx.fig_dir when given a name.
"""
import re

import matplotlib.pyplot as plt
import numpy as np

from .. import fsio
from .swap import AUC_ROWS, FAM_ROWS, GROUPS, RAW, Ctx, binned

MEASURE_ORDER = ['representation', '|risk change|', 'outcome pull']
PALETTE = [('#eb6834', 'o'), ('#2a78d6', 's'), ('#1baf7a', 'D'), ('#8a5cc2', '^'), ('#b8860b', 'v'),
           ('#52514e', 'P')]


def ftag(s):
    return re.sub(r'[^0-9A-Za-z]+', '_', s).strip('_')


def clf_tag(clf):
    return 'raw' if clf == RAW else clf.split()[0].lower()


def _finish(ctx, fig, name, show):
    if name and ctx.fig_dir:
        print('saved', fsio.savefig(fig, f'{ctx.fig_dir}/{name}.png'))
    plt.show() if show else plt.close(fig)
    return fig


def reference_chart(ctx: Ctx, rs, readout, clf, name=None, show=False):
    """Arrows from each group's reference risk to its risk after swapping the signal or the note."""
    t = rs[(rs.readout == readout) & (rs.clf == clf)]
    if t.empty:
        return None
    style = ctx.slot_style
    levels = [l for l in ['L1', 'L2', 'L3'] if l in set(t.level_id)]
    rows = [(f, g) for f in FAM_ROWS for g, _ in GROUPS
            if not t[(t.family == f) & (t.group == g)].empty][::-1]
    slots = [s for s in style if s in set(t.slot)]
    off = {s: (i - (len(slots) - 1) / 2) * 0.28 for i, s in enumerate(slots)}
    vals = np.r_[t.ref, t.swapped - t.delta_ci, t.swapped + t.delta_ci]
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    pad = 0.06 * max(hi - lo, 1e-6)
    fig, axes = plt.subplots(1, len(levels), figsize=(5.8 * len(levels), 0.42 * len(rows) * len(slots) + 1.8),
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
                col = style[s][1]
                y = yi + off[s]
                ax.plot([r.ref], [y], marker='|', ms=11, mew=2, color='#7f7f7f', zorder=2)
                ax.annotate('', xy=(r.swapped, y), xytext=(r.ref, y), zorder=3,
                            arrowprops=dict(arrowstyle='-|>', color=col, lw=1.8, shrinkA=0, shrinkB=0,
                                            mutation_scale=11))
                ax.errorbar(r.swapped, y, xerr=r.delta_ci, fmt='none', ecolor=col, elinewidth=1,
                            capsize=2.5, alpha=0.8, zorder=3)
                ax.text(1.02, y, f'Δ {r.delta:+.3f} ± {r.delta_ci:.3f}  n={r.n}',
                        transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=7, color=col)
        ax.set_xlim(lo - pad, hi + pad)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_title(lv, loc='left'); ax.grid(False)
        ax.set_xlabel('mean P(Yes)' if clf == RAW else 'mean predicted 1-year risk')
    lab = {'label-specific': 'label-specific (from matched)'}
    axes[0][0].set_yticks(range(len(rows)), [f'{lab.get(f, f)} · {g}' for f, g in rows])
    axes[0][0].set_ylim(-0.6, len(rows) - 0.4)
    handles = [plt.Line2D([], [], color=style[s][1], lw=2, marker='>', ms=6, label=style[s][0]) for s in slots]
    handles.append(plt.Line2D([], [], color='#7f7f7f', marker='|', ls='', ms=10, mew=2,
                              label='reference: own baseline records (arrow head = after the swap)'))
    fig.legend(handles=handles, ncol=len(handles), loc='lower left', bbox_to_anchor=(0.01, 0.0),
               frameon=False, fontsize=9)
    how = "the model's own P(Yes), no probe" if clf == RAW else 'probe fitted on the reference only'
    fig.suptitle(f'{ctx.model_name} · {readout} · {clf}: how far each swap moves predicted risk from the '
                 f'reference, {ctx.sig_name} vs note\npaired Δ per patient, {how} · '
                 'counterfactual should push decedents left and survivors right', x=0.01, ha='left', fontsize=10.5)
    h = fig.get_figheight()
    fig.tight_layout(rect=[0, 0.55 / h, 1, 1 - 0.75 / h])
    return _finish(ctx, fig, name, show)


def auroc_chart(ctx: Ctx, au, readout, clf, name=None, show=False):
    """AUROC at the reference and after each swap: signal swap above, note swap below, CIs only."""
    t = au[(au.readout == readout) & (au.clf == clf)]
    if t.empty:
        return None
    style = ctx.slot_style
    levels = [l for l in ['L1', 'L2', 'L3'] if l in set(t.level_id)]
    fams = [f for f in AUC_ROWS if f in set(t.family)][::-1]
    slots = [k for k in style if k in set(t.slot)]
    off = {k: ((len(slots) - 1) / 2 - i) * 0.24 for i, k in enumerate(slots)}     # first slot on top
    lo = min(0.45, np.nanmin(np.r_[t.auroc_ref, t.swap_lo]))
    hi = max(0.55, np.nanmax(np.r_[t.auroc_ref, t.swap_hi]))
    pad = 0.04 * (hi - lo)
    fig, axes = plt.subplots(1, len(levels), figsize=(6.6 * len(levels), 0.95 * len(fams) + 1.6),
                             sharey=True, squeeze=False)
    for ax, lv in zip(axes[0], levels):
        d = t[t.level_id == lv]
        ax.axvline(0.5, color='#bdbdbd', ls='--', lw=1, zorder=0)
        for y, f in enumerate(fams):
            r = d[d.family == f]
            if r.empty:
                continue
            for q in r.itertuples():
                lab, col, mk = style[q.slot]
                yy = y + off[q.slot]
                ax.plot([q.auroc_ref], [yy], marker='|', ms=11, mew=2, color='#7f7f7f', zorder=2)
                ax.errorbar(q.auroc_swap, yy, xerr=[[q.auroc_swap - q.swap_lo], [q.swap_hi - q.auroc_swap]],
                            fmt=mk, color=col, ms=6, mec='white', elinewidth=1.4, capsize=3, zorder=4)
                ax.text(1.02, yy, f'Δ {q.d_auroc:+.3f} [{q.d_lo:+.3f}, {q.d_hi:+.3f}]',
                        transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=7, color=col)
            ax.text(0.01, y + 0.47, f'n={r.n.iloc[0]} ({r.died.iloc[0]} died)', transform=ax.get_yaxis_transform(),
                    fontsize=7, color='#52514e', va='top')
            if y:
                ax.axhline(y - 0.5, color='#d9d8d3', lw=1, zorder=0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(lv, loc='left'); ax.grid(False)
        ax.set_xlabel('AUROC (1-year mortality)')
    axes[0][0].set_yticks(range(len(fams)), fams)
    axes[0][0].set_ylim(-0.5, len(fams) - 0.5)
    handles = [plt.Line2D([], [], color=style[s][1], lw=1.4, marker=style[s][2], ms=6,
                          label=f'{style[s][0]}: AUROC and 95% CI') for s in slots]
    handles.append(plt.Line2D([], [], color='#7f7f7f', marker='|', ls='', ms=12, mew=2.5,
                              label='reference AUROC: own baseline records'))
    fig.legend(handles=handles, ncol=len(handles), loc='lower left', bbox_to_anchor=(0.01, 0.0),
               frameon=False, fontsize=9)
    how = "the model's own P(Yes), no probe" if clf == RAW else 'probe fitted on the reference only'
    fig.suptitle(f'{ctx.model_name} · {readout} · {clf}: AUROC at the reference and after each swap, '
                 f'{ctx.sig_name} vs note\nsame patients and reference for both lines of a row, {how} · '
                 'donor rows: compare blue with orange, not the distance to the reference',
                 x=0.01, ha='left', fontsize=10.5)
    h = fig.get_figheight()
    fig.tight_layout(rect=[0, 0.4 / h, 1, 1])
    return _finish(ctx, fig, name, show)


def temporal_chart(ctx: Ctx, tp, readout, clf='Logistic regression', show=False):
    t = tp[tp.readout == readout]
    if t.empty:
        return None
    levels = [l for l in ['L1', 'L2', 'L3'] if l in set(t.level_id)]
    style = {'sig': (f'{ctx.sig_name} (follow-up record)', '#2a78d6', 'o'),
             'n': ('note (follow-up report)', '#eb6834', 's')}
    fig, axes = plt.subplots(2, len(levels), figsize=(5.4 * len(levels), 7.6), squeeze=False, sharey='row')
    for j, lv in enumerate(levels):
        d = t[t.level_id == lv]
        ax = axes[0][j]
        ax.axhline(0, color='grey', ls='--', lw=1, zorder=1)
        for m, (lab, col, mk) in style.items():
            b = binned(d[f'sim_{m}'], d[f'pull_{m}'])
            ax.errorbar(b.x, b.m, yerr=b.ci, fmt=mk + '-', color=col, ms=6, lw=2, capsize=3,
                        markeredgecolor='white', label=lab, zorder=3)
        ax.set_title(f'{lv} · {len(d)} paired patients, {int(d.y.sum())} died', loc='left')
        ax.set_xlabel('cosine similarity, baseline vs follow-up (1 = unchanged)')
        ax.invert_xaxis(); ax.grid(False)
        ax = axes[1][j]
        ax.axhline(0, color='grey', ls='--', lw=1, zorder=1)
        b = binned(d.sim_mean, d.pull_diff)
        ax.errorbar(b.x, b.m, yerr=b.ci, fmt='D-', color='#52514e', ms=6, lw=2, capsize=3,
                    markeredgecolor='white', zorder=3)
        for x, n in zip(b.x, b.n):
            ax.text(x, 0.03, f'n={n}', transform=ax.get_xaxis_transform(), ha='center', fontsize=7, color='#52514e')
        ax.set_xlabel('mean cosine similarity of the two follow-ups (1 = unchanged)')
        ax.invert_xaxis(); ax.grid(False)
    axes[0][0].set_ylabel('outcome pull\n(Δ risk toward the true outcome)')
    axes[1][0].set_ylabel(f'paired difference\nnote pull − {ctx.sig_name} pull')
    axes[0][0].legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1.1), ncol=2, fontsize=9)
    fig.suptitle(f'{ctx.model_name} · {readout} · {clf}: temporal swap, {ctx.sig_name} vs note on the same patients\n'
                 'top: does the follow-up record move risk the right way, by how much it changed the '
                 'readout · bottom: > 0 the note carries more of it', x=0.01, ha='left', fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return _finish(ctx, fig, f'swap4_temporal_paired_{ftag(readout)}_{clf_tag(clf)}', show)


def dominance_chart(ctx: Ctx, paired, clf, show=False):
    t = paired[paired.clf.isin([clf, RAW])]
    if t.empty:
        return None
    levels = [l for l in ['L1', 'L2', 'L3'] if l in set(t.level_id)]
    readouts = list(dict.fromkeys(t.readout))
    style = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(readouts)}
    rows = [(f, m) for m in MEASURE_ORDER for f in FAM_ROWS
            if not t[(t.family == f) & (t.measure == m)].empty][::-1]
    step = 0.8 / max(len(readouts), 1)
    off = {r: (i - (len(readouts) - 1) / 2) * step for i, r in enumerate(readouts)}
    fig, axes = plt.subplots(1, len(levels),
                             figsize=(5.6 * len(levels), 0.25 * len(rows) * max(len(readouts), 2.5) + 2.2),
                             sharey=True, squeeze=False)
    for ax, lv in zip(axes[0], levels):
        d = t[t.level_id == lv]
        ax.axvline(0, color='grey', ls='--', lw=1, zorder=1)
        for yi, (f, m) in enumerate(rows):
            if yi and rows[yi - 1][1] != m:
                ax.axhline(yi - 0.5, color='#d9d8d3', lw=1, zorder=0)
            for ro in readouts:
                r = d[(d.family == f) & (d.measure == m) & (d.readout == ro)]
                if r.empty:
                    continue
                r = r.iloc[0]
                col, mk = style[ro]
                y = yi + off[ro]
                ax.errorbar(r.dominance, y, xerr=[[r.dominance - r.dominance_lo], [r.dominance_hi - r.dominance]],
                            fmt=mk, color=col, ms=6, capsize=2, lw=1.2, markeredgecolor='white', zorder=3)
                ax.text(1.04, y, f'{r.dominance:+.2f}', transform=ax.get_yaxis_transform(), ha='left',
                        va='center', fontsize=6.5, color=col)
        ax.set_xlim(-1.05, 1.05)
        ax.set_title(lv, loc='left'); ax.grid(False)
        ax.set_xlabel(f'← {ctx.sig_name} dominates     dominance     note dominates →')
    axes[0][0].set_yticks(range(len(rows)), [f'{f} · {m}' for f, m in rows])
    axes[0][0].set_ylim(-0.6, len(rows) - 0.4)
    raw = set(t.readout[t.clf == RAW])
    handles = [plt.Line2D([], [], marker=style[r][1], ls='', color=style[r][0], ms=7,
                          label=f'{r} (raw, no probe)' if r in raw else r) for r in readouts]
    fig.legend(handles=handles, ncol=min(len(handles), 4), loc='lower left', bbox_to_anchor=(0.01, 0.0),
               frameon=False, fontsize=9)
    fig.suptitle(f'{ctx.model_name} · paired {ctx.sig_name} vs note dominance within each swap criterion · '
                 f'vector readouts via {clf}, P(Yes) raw\n'
                 f'(note − {ctx.sig_name}) / (|note| + |{ctx.sig_name}|), same patients and donors, '
                 'paired bootstrap 95% CI', x=0.01, ha='left', fontsize=11)
    h = fig.get_figheight()
    fig.tight_layout(rect=[0, 0.55 / h, 1, 1 - 0.75 / h])
    return _finish(ctx, fig, f'swap4_paired_dominance_{clf_tag(clf)}', show)
