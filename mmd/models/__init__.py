"""Step 2, run: one adapter per model, all following the contract in base.py.

    base.py        Adapter (what a model must implement), Sample + SampleBuilder (record id -> model input),
                   ReadoutSpec, the registry (MODULES, register, get, all_models)
    hf_vlm.py      a generic adapter for Hugging Face image-text models; medgemma.py subclasses it
    chexagent.py, llava_med.py, opentslm.py (Llama + Gemma)   one file per model family

Each module registers its adapter; `base.get(name)` imports it on demand, and torch / model libraries load only
in an adapter's load(), so importing mmd never needs a GPU environment.
"""
