# Bundled source

This directory is the subset of [Vista4D](https://github.com/Eyeline-Labs/Vista4D)
(Apache-2.0, see `LICENSE`) that the geometry service uses for reconstruction and
segmentation: the Pi3X reconstruction path (`pi3/`, `utils/recon_and_seg/`) and SAM3
segmentation (`sam3/`), driven by `scripts/preprocess/recon_and_seg_single.py`. The video
diffusion part of Vista4D is not included.

Local changes, all outside the models' numerics:

- numerical guards in the Pi3X windowed (VO) path, including a 32-bit index limit;
- dense per-frame RGB-D, camera and dynamic-mask export, plus per-instance SAM3 tracks;
- per-phase timing and peak GPU memory reporting.

Bundled subprojects keep their upstream licenses: Pi3 ([yyfz/Pi3](https://github.com/yyfz/Pi3))
and SAM3 ([facebookresearch/sam3](https://github.com/facebookresearch/sam3)). Weights are not
included; see the top-level README.
