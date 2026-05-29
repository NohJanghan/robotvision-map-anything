# Outputs Directory

Generated reconstruction results, renders, metrics, and logs are excluded from Git.
The placeholders in this directory document the expected experiment layout.

MapAnything configurations:

- `config_a_uncalibrated_image_only`: raw RGB images only
- `config_b_calibrated_image_only`: RGB images plus COLMAP intrinsics
- `config_c_calibrated_colmap_poses`: RGB images plus COLMAP intrinsics and poses
- `config_d_calibrated_ar_poses`: optional ARKit/ARCore pose input
