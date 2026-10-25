"""Step 3, analyse: one model's stored embeddings -> effect tables and charts, the same for every model (CPU only).

    probe.py     Embeddings (record ids -> feature matrix) and Probe (fit on the reference embeddings per fold,
                 score every perturbed input) -> one long table: one row per (stay, configuration)
    effects.py   every table computed from that long table: Δ risk by outcome, AUROC, within-patient swaps,
                 reliance shares per modality, pairwise dominance, informativeness, the label-shuffle test
    stats.py     helpers: Wilcoxon, stratified bootstrap, rank AUROC
    charts.py    the figures drawn from the effect tables
    report.py    analyse_model(): probe -> effects -> charts, and writes the tables (called by zoo.Zoo.analyse)
"""
