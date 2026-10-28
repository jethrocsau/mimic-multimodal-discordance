"""Qwen2.5-VL-7B-Instruct: a general VLM with no MIMIC-CXR training. Prefix fusion: the ViT's patches are merged
2x2 by `visual.merger` into LLM tokens placed at the image pad tokens.

Differences from the Gemma-style models in hf_vlm:
- the image token count depends on resolution, so every film is resized to about the same pixel count
  (FIXED_PIXELS, aspect ratio kept): ~256 merged tokens, varying a little with the film's shape;
- the processor takes a flat image list, and the vision features come out flat (all images' tokens, H), split
  per image by image_grid_thw (image_token_counts);
- `visual` already ends in the merger, so the encoder readout is skipped (it would equal the projector's);
- no mean swap yet: the merger needs image_grid_thw, so a cohort-mean projector output is not wired in.
"""
from .base import register
from .hf_vlm import HFImageTextAdapter, find_module

FIXED_PIXELS = 448 * 448             # ~16x16 merged tokens per film (aspect ratio kept)


@register
class Qwen25VL7B(HFImageTextAdapter):
    name = 'qwen2.5-vl-7b'
    hf_id = 'Qwen/Qwen2.5-VL-7B-Instruct'
    revision = 'main'                 # pin a commit once the smoke test passes
    signal, modalities, default_modset = 'cxr', ('x', 'n'), 'cxr_notes'
    meta = dict(fusion='prefix (patch merger)', domain='general', params='7B', mimic_exposure='none')
    supports_mean = False

    def load(self, device='cuda'):
        super().load(device)
        ip = self.proc.image_processor
        ip.min_pixels = ip.max_pixels = FIXED_PIXELS
        if hasattr(ip, 'size') and isinstance(ip.size, dict):
            ip.size = dict(ip.size, shortest_edge=FIXED_PIXELS, longest_edge=FIXED_PIXELS)
        self.vt_name, self.vt = None, None
        self.pj_name, self.pj = find_module(self.model, 'visual.merger')
        return self

    def batch_images(self, images):
        return list(images)               # flat: one image per prompt, matched to the image pads in order

    def image_token_counts(self, x):
        merge = self.model.config.vision_config.spatial_merge_size
        return (x['image_grid_thw'].prod(-1) // merge ** 2).tolist()

    def inputs(self, sample):
        kw = dict(text=[self.prompt(sample)], return_tensors='pt', add_special_tokens=False)
        if sample.signal is not None:
            kw['images'] = [sample.signal]
        x = self.proc(**kw).to(self.dev)
        if 'pixel_values' in x:
            x['pixel_values'] = x['pixel_values'].to(self.model.dtype)
        return x
