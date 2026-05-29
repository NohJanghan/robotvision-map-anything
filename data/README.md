# Data Directory

Local datasets are intentionally excluded from Git. Keep RGB sequences, AR captures,
and converted inputs here when running COLMAP and MapAnything.

Expected layout:

```text
data/
  raw/
    rgb_sequences/        # original RGB image sequences
    ar_captures/          # optional ARKit/ARCore capture exports
  processed/
    colmap_exports/       # cameras.txt, images.txt, points3D.txt copied from COLMAP
    mapanything_inputs/   # inputs converted for MapAnything configurations
```
