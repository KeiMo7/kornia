`kornia.contrib.PanoramaStitcher` stitches ordered images into a panorama with overlap-only per-channel exposure
compensation and distance-feather blending, returning a validity mask, seam-quality metrics, review slices and
explicit rejection of pairs with too few inliers or too small an overlap. The functional helpers
`boundary_distance`, `estimate_pair_gain`, `feather_weights`, `fuse_panorama`, `seam_discontinuity` and
`overlap_luma_difference` are exported alongside it; the existing `ImageStitcher` behavior is unchanged.
