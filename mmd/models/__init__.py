"""Model adapters. Each module registers one adapter; `mmd.models.base.get(name)` imports it on demand, so
importing mmd never imports torch or a model library."""
