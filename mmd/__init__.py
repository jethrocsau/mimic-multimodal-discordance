"""mmd: modality discordance across a zoo of multimodal models, on one frozen MIMIC-IV cardiac ICU cohort.

How much does each model's 1-year mortality answer rely on each modality (CXR image, ECG, note, ...), compared with
how much that modality tells it on its own? Every model gets the same perturbed inputs; the answer is one table.

The package runs in four steps; `zoo.Zoo` drives all of them for every model, and `cli.py` exposes each as a command.

    1. define    modsets.py   a modality set: a name + modality keys, e.g. cxr_ecg_notes = (e, x, n)
                 records.py   the record id: one string naming one complete model input
                 data.py      phase0 snapshot; which record fills each stay's first / last slot; loaders
                 cohort.py    TierCohort (recipients + folds), DonorPlan, Manifest (frozen, hashed)
                 perturb.py   the perturbation strategies -> every perturbed record id -> the manifest
    2. run       models/      one Adapter per model: Sample -> readouts (P(Yes), hidden states)
                 runner.py    the manifest's missing records -> adapter -> store (GPU)
                 store.py     EmbeddingStore: readouts by record id, its index, run log
    3. analyse   analysis/    probe -> effect tables -> charts, per model and modality set (CPU)
    4. compare   zoo.py       information vs reliance per modality, across models; orchestrates 1-3

    shared       config.py (settings, paths under `root`), fsio.py (local / gs:// IO), prompts.py (question text)

Three things tie the steps together:
    the record id      the manifest row, the store key, the resume check and the probe lookup are one string
    the manifest hash  every model runs and is analysed on exactly the same inputs (checked at run and analyse)
    the adapter        analysis never sees a model, only named arrays; a new model is one new adapter

See mmd/README.md for setup and the commands, dev/architecture.md for the design.
"""
__version__ = "0.1.0"
