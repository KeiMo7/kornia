Image Stitching
===============

.. meta::
   :description: Create image panoramas with Kornia's PyTorch ImageStitcher, LoFTR matching and RANSAC estimation.

Image stitching is the process of combining multiple images with overlapping fields of view to produce a panorama. Here, we provide :py:class:`~kornia.contrib.ImageStitcher` to easily stitch a number of images.

.. image:: https://raw.githubusercontent.com/kornia/data/main/matching/stitch_before.png
   :alt: Input images to be stitched

Learn more: https://paperswithcode.com/task/image-stitching/

.. code:: python

    import torch
    import kornia.feature as KF
    from kornia.contrib import ImageStitcher
    from kornia.io import ImageLoadType, load_image

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    imgs = [load_image(p, ImageLoadType.RGB32, device=device)[None] for p in ("left.jpg", "right.jpg")]

    matcher = KF.LoFTR(pretrained="outdoor")
    stitcher = ImageStitcher(matcher, estimator="ransac").to(device)
    # NOTE: stitching many images at once can require a lot of memory.
    with torch.no_grad():
        panorama = stitcher(*imgs)  # (1, 3, H, W_pano)

.. image:: https://raw.githubusercontent.com/kornia/data/main/panorama/out_panorama.jpg
   :alt: Stitched panorama

Explore with your data: https://colab.research.google.com/github/kornia/tutorials/blob/master/source/image_stitching.ipynb

Seam-aware stitching with exposure compensation
------------------------------------------------

:py:class:`~kornia.contrib.ImageStitcher` pastes each warped image over the panorama so far, which leaves a hard seam
when two views differ in exposure or white balance. :py:class:`~kornia.contrib.PanoramaStitcher` keeps the same
matcher and homography machinery but renders every image onto one canvas, estimates one per-channel exposure gain per
image from the overlap pixels only, and fuses with weights derived from each image's distance to its own coverage
boundary. It returns a validity mask and, with ``return_details=True``, seam-quality metrics and before/after crops
for manual review. Pairs with too few inliers or too small an overlap are rejected with a
:py:class:`~kornia.contrib.StitchingError` that names the pair and its counts instead of producing a malformed
panorama.

.. code:: python

    import torch
    import kornia.feature as KF
    from kornia.contrib import PanoramaStitcher
    from kornia.io import ImageLoadType, load_image

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    imgs = [load_image(p, ImageLoadType.RGB32, device=device)[None] for p in ("left.jpg", "middle.jpg", "right.jpg")]

    matcher = KF.LocalFeatureMatcher(KF.SIFTFeature(200), KF.DescriptorMatcher("snn", 0.8))
    stitcher = PanoramaStitcher(matcher, estimator="ransac", min_inliers=8, min_overlap_fraction=0.05).to(device)
    with torch.no_grad():
        panorama, valid_mask, quality, seam_slices = stitcher(imgs, reference_index=0, return_details=True)

    # Fusion only, with geometry supplied from outside (the matcher is never called):
    homographies, inlier_counts = stitcher.estimate_homographies(imgs)
    panorama, valid_mask = stitcher.fuse(imgs, homographies)
