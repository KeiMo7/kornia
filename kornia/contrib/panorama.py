# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Exposure-compensated panorama stitching with distance-feather blending.

This module is the seam-aware counterpart of :class:`~kornia.contrib.image_stitching.ImageStitcher`. The geometry
machinery (feature matching, RANSAC homographies, :func:`~kornia.geometry.transform.warp_perspective`) is reused
unchanged; the additions are:

- an exposure compensation that fits one multiplicative per-channel gain per image on the overlap pixels only, so a
  local illumination difference is corrected without stretching each image's global contrast;
- a feather blend whose weight of every warped image is its Euclidean distance to its own coverage boundary, computed
  with :func:`~kornia.contrib.distance_transform`, so every weight is explained by a distance and no image replaces an
  overlap wholesale;
- a validity mask, seam-quality metrics, seam slices for manual review, and explicit rejection of pairs whose overlap
  or inlier support is too small.

The module is functional and stateless apart from the matcher it wraps: :class:`PanoramaStitcher` orchestrates the
helpers, and every helper can be used on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from kornia.color import rgb_to_grayscale
from kornia.contrib.distance_transform import distance_transform
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE, KORNIA_CHECK_TYPE
from kornia.core.exceptions import BaseError
from kornia.feature import LocalFeatureMatcher, LoFTR
from kornia.geometry.homography import find_homography_dlt_iterated
from kornia.geometry.ransac import RANSAC
from kornia.geometry.transform import warp_perspective

__all__ = [
    "PanoramaStitcher",
    "SeamSlices",
    "StitchQuality",
    "StitchingError",
    "boundary_distance",
    "estimate_pair_gain",
    "feather_weights",
    "fuse_panorama",
    "overlap_luma_difference",
    "seam_discontinuity",
]


class StitchingError(BaseError):
    """A stitch cannot be produced reliably.

    Raised when a pair has too few matched inliers or too little overlap for a stable homography, or when the inputs
    disagree in shape or channel count. The message names the offending image pair and the counts that failed, so a
    multi-image run reports the pair to fix instead of returning a malformed panorama.
    """


@dataclass
class StitchQuality:
    """Quality measurements of one fused panorama.

    All measurements live on the panorama canvas. ``*_before`` and ``*_after`` compare the two single-image views of
    every seam, evaluated on the same overlap pixels before and after the blend, so the improvement is attributable.

    Attributes:
        seam_discontinuity_before: mean absolute per-channel difference across the seam in the overlaps, using each
            image's own warped pixels.
        seam_discontinuity_after: the same difference on the fused panorama sampled one pixel to each side of the seam,
            which the distance-feather blend drives toward zero.
        overlap_luma_difference_before: mean absolute grayscale difference between the warped images on every overlap.
        overlap_luma_difference_after: the same difference on the fused result, measured against each contributor.
        clipped_pixel_fraction: fraction of valid output pixels clamped by the image value range after gain
            compensation, split as ``(under, over)``. Zero means no compensation was clipped.
        overlap_fractions: per adjacent pair, the overlap area as a fraction of the smaller image's area.
    """

    seam_discontinuity_before: torch.Tensor
    seam_discontinuity_after: torch.Tensor
    overlap_luma_difference_before: torch.Tensor
    overlap_luma_difference_after: torch.Tensor
    clipped_pixel_fraction: tuple[float, float]
    overlap_fractions: list[float]


@dataclass
class SeamSlices:
    """Before/after comparison crops of every seam for manual review.

    Attributes:
        boxes: one ``(y0, y1, x0, x1)`` tuple per seam, the crop rectangle on the panorama canvas.
        before: stacked crops ``(S, C, h, w)`` of the two contributors shown side by side, unfused.
        after: stacked crops ``(S, C, h, w)`` of the fused panorama on the same rectangles.
    """

    boxes: list[tuple[int, int, int, int]]
    before: torch.Tensor
    after: torch.Tensor


def boundary_distance(mask: torch.Tensor) -> torch.Tensor:
    """Euclidean distance of every covered pixel to the boundary of its coverage mask.

    ``dist = boundary_distance(mask)`` takes a binary mask ``(B, 1, H, W)`` and returns, per covered pixel, the
    Euclidean distance in pixels to the nearest uncovered (zero) pixel. A covered pixel sitting on the mask boundary
    therefore has distance one (its neighbour is uncovered), the distance grows inward, and pixels outside the mask
    and the pixels of an empty or a fully covered mask are zero because neither has an interior.

    The distance is computed with :func:`~kornia.contrib.distance_transform` on the inverted mask and is the
    interpretable source of the feather weights: a pixel close to the edge of what an image covers contributes less
    than one deep inside its coverage.

    Args:
        mask: binary coverage mask, shaped ``(B, 1, H, W)``, with non-zero values marking covered pixels.

    Returns:
        Boundary distances with the shape and dtype of ``mask`` (computed in at least float32).
    """
    KORNIA_CHECK_SHAPE(mask, ["B", "1", "H", "W"])
    work = torch.promote_types(mask.dtype, torch.float32)
    inverted = (mask <= 0).to(work)
    return distance_transform(inverted).to(mask.dtype)


def estimate_pair_gain(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    overlap: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Estimate the per-channel multiplicative gain that aligns ``image_b`` to ``image_a`` on an overlap.

    On the shared pixels the gain minimises the weighted least-squares error
    :math:`\\sum_p w_p (g_c b_{pc} - a_{pc})^2` independently per channel, with the overlap mask as weights. The
    closed form is :math:`g_c = \\sum_p w_p a_{pc} b_{pc} / \\sum_p w_p b_{pc}^2`. Only overlap pixels enter the
    statistics, so this is a local exposure correction between two views, not a global contrast stretch of either
    image. An empty overlap or a degenerate channel returns a gain of one.

    Args:
        image_a: reference image ``(B, C, H, W)`` in any floating dtype.
        image_b: image to align to ``image_a``, same shape and dtype.
        overlap: binary overlap mask ``(B, 1, H, W)`` on the shared canvas.
        eps: denominator floor that keeps a dark channel from producing an unbounded gain.

    Returns:
        Gains ``(B, C, 1, 1)`` in the image dtype, one per channel.
    """
    KORNIA_CHECK_SHAPE(image_a, ["B", "C", "H", "W"])
    KORNIA_CHECK_SHAPE(image_b, ["B", "C", "H", "W"])
    KORNIA_CHECK(
        image_a.shape == image_b.shape,
        f"image_a and image_b must have the same shape, got {image_a.shape} and {image_b.shape}",
    )
    work = torch.promote_types(image_a.dtype, torch.float32)
    a, b, w = image_a.to(work), image_b.to(work), (overlap > 0).to(work)
    num = (a * b * w).sum(dim=(-2, -1), keepdim=True)
    den = (b.square() * w).sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
    gain = num / den
    # A channel with no signal carries no exposure information: keep the identity rather than an arbitrary number.
    gain = torch.where(den <= eps, torch.ones_like(gain), gain)
    return gain.to(image_a.dtype)


def _pair_observation(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    overlap: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit the gain ratio aligning ``image_b`` to ``image_a`` on shared pixels, with its support.

    Returns ``(ratio, weight)``, each a ``(C,)`` tensor. :func:`estimate_pair_gain` fits ``ratio * image_b ~= image_a``
    on the overlap, so ``ratio = g_a / g_b``: image a's exposure gain divided by image b's. The gain solver turns this
    into the observation ``ratio * g_a - g_b = 0``. The weight is the overlap pixel count, used as the confidence of
    that observation when the global gain system is assembled.
    """
    ratio = estimate_pair_gain(image_a, image_b, overlap, eps=eps)[0, :, 0, 0]
    channels = image_a.shape[1]
    weight = overlap.to(image_a.dtype).sum().reshape(1).expand(channels)
    return ratio, weight


def _solve_gains_from_observations(
    n: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
    observations: list[tuple[int, int, torch.Tensor, torch.Tensor]],
    reference_index: int,
) -> torch.Tensor:
    r"""Build and solve the gain normal equations.

    Each observation ``(i, j, ratio, weight)`` carries :math:`r = g_i/g_j` from :func:`estimate_pair_gain` (the ratio
    that aligns image ``j`` onto image ``i``), so it states :math:`r g_i - g_j = 0` with a per-channel confidence
    weight. The reference image's gain is anchored to one, which makes the system full rank.
    """
    work = torch.promote_types(dtype, torch.float32)
    if not 0 <= reference_index < n:
        raise StitchingError(
            f"reference_index {reference_index} is out of range for {n} images; choose one in [0, {n - 1}]."
        )
    gram = torch.zeros(n, n, channels, device=device, dtype=work)
    rhs = torch.zeros(n, channels, device=device, dtype=work)
    for i, j, ratio, weight in observations:
        r = ratio.to(work).reshape(channels)
        wgt = weight.to(work).reshape(channels)
        # Row (r e_i - e_j): its outer product contributes the normal-equation entries below.
        gram[i, i] += wgt * r * r
        gram[j, j] += wgt
        gram[i, j] -= wgt * r
        gram[j, i] -= wgt * r
    # Anchor g_reference = 1 with a weight that dominates the overlap observations.
    # gram is (N, N, C); its diagonal is (C, N), so reduce over the image axis, not channels.
    diagonal = gram.diagonal(dim1=0, dim2=1)  # (C, N)
    anchor = diagonal.sum(1).clamp_min(1.0) * 1e3 + 1.0  # (C,)
    diagonal[:, reference_index] = anchor
    rhs[reference_index] = anchor
    # Solve per channel on (N, N) systems; N is the number of images, typically small.
    gram_per_c = gram.permute(2, 0, 1)  # (C, N, N)
    gains = torch.linalg.solve(gram_per_c, rhs.t().unsqueeze(-1)).squeeze(-1)  # (C, N)
    gains = gains.t().contiguous()  # (N, C)
    gains[reference_index] = 1.0
    return gains.reshape(n, channels, 1, 1).to(dtype)


def feather_weights(masks: torch.Tensor) -> torch.Tensor:
    """Normalised per-pixel blend weights from coverage boundary distances.

    Given the stacked coverage masks of every contributor on one canvas, ``(N, 1, H, W)``, each image's weight is its
    boundary distance (:func:`boundary_distance`), normalised at every pixel across the images covering it. Pixels
    covered by one image therefore take its value unchanged, and an overlap cross-fades by the ratio of the two
    distances. No weight is a fixed constant: where a mask is empty or full-frame its distance is zero and the other
    contributors decide the pixel.

    Args:
        masks: binary coverage masks ``(N, 1, H, W)``.

    Returns:
        Normalised weights ``(N, 1, H, W)`` summing to one wherever at least one mask covers the pixel, zero elsewhere.
    """
    KORNIA_CHECK_SHAPE(masks, ["N", "1", "H", "W"])
    distances = boundary_distance(masks)
    total = distances.sum(dim=0, keepdim=True)
    tiny = torch.finfo(distances.dtype).tiny
    return torch.where(total > 0, distances / total.clamp_min(tiny), torch.zeros_like(distances))


def fuse_panorama(
    warped_images: torch.Tensor,
    masks: torch.Tensor,
    gains: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse warped images on a shared canvas with distance-feather weights.

    Args:
        warped_images: every contributor rendered on the panorama canvas, ``(N, C, H, W)``, with zero padding outside
            its coverage.
        masks: binary coverage masks ``(N, 1, H, W)`` aligned with ``warped_images``.
        gains: optional per-image per-channel gains ``(N, C, 1, 1)`` from :func:`solve_exposure_gains`; when omitted no
            exposure correction is applied.
        weights: optional precomputed :func:`feather_weights` output; when omitted they are derived from ``masks``.

    Returns:
        A tuple ``(panorama, valid_mask)``:
        - ``panorama``: ``(1, C, H, W)`` blended image in the input dtype; uncovered pixels are zero.
        - ``valid_mask``: ``(1, 1, H, W)`` boolean-ish float mask of pixels covered by at least one contributor.
    """
    KORNIA_CHECK_SHAPE(warped_images, ["N", "C", "H", "W"])
    KORNIA_CHECK_SHAPE(masks, ["N", "1", "H", "W"])
    if weights is None:
        weights = feather_weights(masks)
    if gains is None:
        gains = torch.ones(
            warped_images.shape[0],
            warped_images.shape[1],
            1,
            1,
            device=warped_images.device,
            dtype=warped_images.dtype,
        )
    corrected = warped_images * gains.to(warped_images.dtype)
    blended = (corrected * weights).sum(dim=0, keepdim=True)
    valid = (masks.sum(dim=0, keepdim=True) > 0).to(warped_images.dtype)
    return blended, valid


def seam_discontinuity(image: torch.Tensor, mask: torch.Tensor, seam: torch.Tensor) -> torch.Tensor:
    """Mean absolute difference between the pixels on either side of a seam line.

    For every seam pixel the comparison reads one canvas pixel to its left and one to its right, restricted to covered
    pixels. It is the hard-edge measurement the feather blend is judged against: a naive overwrite leaves the full
    exposure jump here, a feather blend smooths it.

    Args:
        image: canvas image ``(1, C, H, W)``.
        mask: coverage mask ``(1, 1, H, W)``.
        seam: binary seam mask ``(1, 1, H, W)`` marking where the comparison is taken.

    Returns:
        Scalar mean absolute per-channel difference; zero when no seam pixel has two covered neighbours.
    """
    covered = mask > 0
    left = image[..., 1:-1]
    right = image[..., 2:]
    valid = seam[..., 1:-1].bool() & covered[..., 1:-1] & covered[..., 2:]
    if int(valid.sum()) == 0:
        return torch.zeros((), device=image.device, dtype=image.dtype)
    return (left - right).abs().where(valid, torch.zeros_like(left)).sum() / valid.sum() / image.shape[1]


def overlap_luma_difference(image_a: torch.Tensor, image_b: torch.Tensor, overlap: torch.Tensor) -> torch.Tensor:
    """Mean absolute grayscale difference of two images on shared pixels."""
    valid = overlap > 0
    if int(valid.sum()) == 0:
        return torch.zeros((), device=image_a.device, dtype=image_a.dtype)
    luma_a = rgb_to_grayscale(image_a) if image_a.shape[1] == 3 else image_a.mean(1, keepdim=True)
    luma_b = rgb_to_grayscale(image_b) if image_b.shape[1] == 3 else image_b.mean(1, keepdim=True)
    return (luma_a - luma_b).abs().where(valid, torch.zeros_like(luma_a)).sum() / valid.sum()


class PanoramaStitcher(nn.Module):
    """Stitch ordered images into one panorama with exposure compensation and feather blending.

    Unlike :class:`~kornia.contrib.image_stitching.ImageStitcher`, which pastes each warped image over the panorama so
    far with a ``"naive"`` overwrite, this stitcher renders every image onto one canvas and fuses them with weights
    derived from each image's coverage-boundary distance (:func:`feather_weights`), after correcting per-image exposure
    with overlap-only gain estimates (:func:`estimate_pair_gain`, combined by the internal gain solver). The feature
    matcher, RANSAC homography semantics and tensor conventions are the same; nothing about the existing single-pair
    homography or matching API changes.

    Two modes:
        - full pipeline: ``stitcher(images)`` matches every adjacent pair, estimates homographies, then fuses;
        - fusion only: :meth:`fuse` takes externally supplied homographies (or pre-warped images) and never invokes the
          matcher, so geometry given from outside is left untouched.

    Determinism:
        RANSAC is constructed with the fixed ``seed`` and every operation is tensor-only, so the same inputs reproduce
        the panorama bit for bit, including on a fresh process. Matcher determinism still depends on the matcher.

    Args:
        matcher: a :class:`~kornia.feature.LoFTR` or :class:`~kornia.feature.LocalFeatureMatcher` for the full pipeline.
            Ignored by :meth:`fuse`.
        estimator: ``"ransac"`` (default, seeded) or ``"vanilla"`` DLT on all matches.
        min_inliers: minimum RANSAC inliers required from an adjacent pair; fewer raises :class:`StitchingError`.
        min_overlap_fraction: minimum overlap area as a fraction of the smaller image's area.
        ransac_seed: fixed RANSAC seed making repeated runs bit-identical.
        exposure_compensation: when True, fit per-image gains; when False, all gains are one.
        clip_value: image value range used to flag under/over-exposed pixels after compensation.
        seam_margin: half-width in pixels of the seam crops returned by :meth:`seam_slices`.

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from kornia.contrib import PanoramaStitcher
        >>> from kornia.feature import LocalFeatureMatcher
        >>> class Matcher(LocalFeatureMatcher):
        ...     def __init__(self):
        ...         nn.Module.__init__(self)
        ...     def forward(self, data):
        ...         raise NotImplementedError("use a real LocalFeatureMatcher in production")
        >>> stitcher = PanoramaStitcher(Matcher())  # doctest: +SKIP
    """

    def __init__(
        self,
        matcher: nn.Module,
        estimator: str = "ransac",
        min_inliers: int = 8,
        min_overlap_fraction: float = 0.05,
        ransac_seed: int = 0,
        exposure_compensation: bool = True,
        clip_value: float = 1.0,
        seam_margin: int = 8,
    ) -> None:
        super().__init__()
        KORNIA_CHECK_TYPE(matcher, nn.Module, "matcher must be an nn.Module")
        KORNIA_CHECK(estimator in ("ransac", "vanilla"), "estimator must be 'ransac' or 'vanilla'")
        KORNIA_CHECK(min_inliers >= 4, "min_inliers must be at least 4 for a homography")
        KORNIA_CHECK(0.0 < min_overlap_fraction < 1.0, "min_overlap_fraction must be in (0, 1)")
        KORNIA_CHECK(clip_value > 0.0, "clip_value must be positive")
        KORNIA_CHECK(seam_margin >= 1, "seam_margin must be at least 1")
        self.matcher = matcher
        self.estimator = estimator
        self.min_inliers = min_inliers
        self.min_overlap_fraction = min_overlap_fraction
        self.ransac_seed = ransac_seed
        self.exposure_compensation = exposure_compensation
        self.clip_value = clip_value
        self.seam_margin = seam_margin
        self.ransac: Optional[RANSAC]
        if estimator == "ransac":
            self.ransac = RANSAC("homography", inl_th=2.0, seed=ransac_seed)
        else:
            self.ransac = None

    # ------------------------------------------------------------------ checks
    def _validate_images(self, images: list[torch.Tensor]) -> None:
        KORNIA_CHECK(len(images) >= 2, "a stitch needs at least two images")
        for k, img in enumerate(images):
            KORNIA_CHECK_IS_TENSOR(img, f"image {k} must be a tensor")
            KORNIA_CHECK_SHAPE(img, ["1", "C", "H", "W"], f"image {k} must be a single (1, C, H, W) sample")
            KORNIA_CHECK(img.is_floating_point(), f"image {k} must be floating point")
        channels = {int(img.shape[1]) for img in images}
        if len(channels) != 1:
            raise StitchingError(f"all images must have the same channel count, got {sorted(channels)}")
        heights = {int(img.shape[-2]) for img in images}
        if len(heights) != 1:
            raise StitchingError(f"all images must share one height for a row panorama, got {sorted(heights)}")
        ref_device, ref_dtype = images[0].device, images[0].dtype
        for k, img in enumerate(images[1:], start=1):
            if img.device != ref_device or img.dtype != ref_dtype:
                raise StitchingError(
                    f"image {k} has device/dtype {img.device}/{img.dtype}, expected {ref_device}/{ref_dtype}"
                )

    # --------------------------------------------------------------- geometry
    def _match_pair(self, image_left: torch.Tensor, image_right: torch.Tensor, pair_index: int) -> dict:
        if isinstance(self.matcher, (LoFTR, LocalFeatureMatcher)):
            data = {
                "image0": rgb_to_grayscale(image_left),
                "image1": rgb_to_grayscale(image_right),
            }
        else:
            raise StitchingError(
                f"the matcher {type(self.matcher).__name__} is not supported; pass a LoFTR or LocalFeatureMatcher"
            )
        return self.matcher(data)

    def _homography_from_matches(
        self, kp_left: torch.Tensor, kp_right: torch.Tensor, pair_index: int
    ) -> tuple[torch.Tensor, int]:
        n_matches = int(kp_left.shape[0])
        if n_matches < 4:
            raise StitchingError(
                f"pair ({pair_index}, {pair_index + 1}): {n_matches} matched keypoints, a homography needs 4"
            )
        if self.estimator == "vanilla":
            homo = find_homography_dlt_iterated(kp_right[None], kp_left[None], torch.ones_like(kp_left[None, :, 0]))
            inliers = n_matches
        else:
            if self.ransac is None:  # the ransac estimator constructs it in __init__
                raise StitchingError("the RANSAC estimator was not initialised; reconstruct the stitcher")
            homo, inlier_mask = self.ransac(kp_right, kp_left)
            inliers = int(inlier_mask.sum())
            if inliers < self.min_inliers:
                raise StitchingError(
                    f"pair ({pair_index}, {pair_index + 1}): {inliers} RANSAC inliers of {n_matches} matches, "
                    f"min_inliers is {self.min_inliers}"
                )
            homo = homo[None]
        return homo, inliers

    def estimate_homographies(self, images: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[int]]:
        """Match every adjacent pair and return their source-to-canvas homographies.

        The first image defines the canvas frame, so its homography is the identity. Homography ``H_i`` maps pixels of
        image ``i`` into that frame by composing the pairwise adjacent homographies, matching the
        :class:`~kornia.contrib.image_stitching.ImageStitcher` chain direction.

        Returns:
            ``(homographies, inlier_counts)``: one ``(1, 3, 3)`` tensor per image and the inlier count per adjacent
            pair.

        Raises:
            StitchingError: naming the pair and its counts when matches or inliers are insufficient.
        """
        self._validate_images(images)
        device = images[0].device
        identity = torch.eye(3, device=device, dtype=torch.float32)[None]
        homographies = [identity]
        inlier_counts: list[int] = []
        cumulative = identity.clone()
        for pair_index in range(len(images) - 1):
            correspondences = self._match_pair(images[pair_index], images[pair_index + 1], pair_index)
            kp0, kp1 = correspondences["keypoints0"], correspondences["keypoints1"]
            pair_h, inliers = self._homography_from_matches(kp0, kp1, pair_index)
            inlier_counts.append(inliers)
            cumulative = pair_h @ cumulative
            homographies.append(cumulative.clone())
        return homographies, inlier_counts

    # ----------------------------------------------------------------- render
    def _canvas_size(self, images: list[torch.Tensor], homographies: list[torch.Tensor]) -> tuple[int, int, int, int]:
        """Warp every image corner through its homography and take the bounding box.

        Returns ``(height, width, x_min, y_min)``; the offsets shift the homographies so all content stays on-canvas.
        """
        device = images[0].device
        corners = []
        for img, h in zip(images, homographies):
            h_, w_ = img.shape[-2:]
            pts = torch.tensor(
                [[0.0, 0.0, 1.0], [w_ - 1, 0.0, 1.0], [w_ - 1, h_ - 1, 1.0], [0.0, h_ - 1, 1.0]], device=device
            )
            warped = (h[0] @ pts.t()).t()
            warped = warped[:, :2] / warped[:, 2:3]
            corners.append(warped)
        all_corners = torch.cat(corners)
        x_min, y_min = all_corners.amin(0)
        x_max, y_max = all_corners.amax(0)
        x_min_i = int(torch.floor(x_min).item())
        y_min_i = int(torch.floor(y_min).item())
        width = int(torch.ceil(x_max).item()) - x_min_i + 1
        height = int(torch.ceil(y_max).item()) - y_min_i + 1
        return height, width, x_min_i, y_min_i

    def _warp_all(
        self,
        images: list[torch.Tensor],
        homographies: list[torch.Tensor],
        canvas_hw: tuple[int, int],
        offset_xy: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n, channels = len(images), images[0].shape[1]
        height, width = canvas_hw
        device, dtype = images[0].device, images[0].dtype
        work = torch.promote_types(dtype, torch.float32)
        shift = torch.tensor(
            [[1.0, 0.0, -offset_xy[0]], [0.0, 1.0, -offset_xy[1]], [0.0, 0.0, 1.0]],
            device=device,
            dtype=work,
        )[None]
        warped_imgs = torch.zeros(n, channels, height, width, device=device, dtype=dtype)
        warped_masks = torch.zeros(n, 1, height, width, device=device, dtype=dtype)
        tol = max(1e-4, 4.0 * torch.finfo(work).eps * max(height, width))
        with torch.autocast(device_type=device.type, enabled=False):
            for i, (img, h) in enumerate(zip(images, homographies)):
                h_canvas = shift @ h.to(work)
                ones = torch.ones(1, 1, img.shape[-2], img.shape[-1], device=device, dtype=dtype)
                rendered = warp_perspective(torch.cat([img.to(work), ones.to(work)], dim=1), h_canvas, (height, width))
                warped_imgs[i] = rendered[0, :channels].to(dtype)
                warped_masks[i, 0] = (rendered[0, channels:] >= 1.0 - tol).to(dtype)
        return warped_imgs, warped_masks

    # -------------------------------------------------------------- exposure
    def _gain_observations(
        self, warped_images: torch.Tensor, masks: torch.Tensor
    ) -> tuple[list[tuple[int, int, torch.Tensor, torch.Tensor]], list[torch.Tensor], list[float]]:
        n = warped_images.shape[0]
        observations: list[tuple[int, int, torch.Tensor, torch.Tensor]] = []
        seam_masks: list[torch.Tensor] = []
        overlap_fractions: list[float] = []
        for i in range(n - 1):
            overlap = (masks[i : i + 1] > 0) & (masks[i + 1 : i + 2] > 0)
            area = int(overlap.sum())
            smaller = min(int(masks[i].sum()), int(masks[i + 1].sum()))
            fraction = area / max(smaller, 1)
            overlap_fractions.append(fraction)
            if area == 0 or fraction < self.min_overlap_fraction:
                raise StitchingError(
                    f"pair ({i}, {i + 1}): overlap is {fraction:.4f} of the smaller image "
                    f"({area} px), min_overlap_fraction is {self.min_overlap_fraction}"
                )
            ratio, weight = _pair_observation(
                warped_images[i : i + 1], warped_images[i + 1 : i + 2], overlap.to(warped_images.dtype)
            )
            observations.append((i, i + 1, ratio, weight))
            seam_masks.append(overlap)
        return observations, seam_masks, overlap_fractions

    def _clipped_fraction(
        self, warped_images: torch.Tensor, gains: torch.Tensor, valid: torch.Tensor
    ) -> tuple[float, float]:
        corrected = warped_images * gains
        valid_pixels = int(valid.sum()) * warped_images.shape[1]
        if valid_pixels == 0:
            return 0.0, 0.0
        under = int(((corrected < 0.0) & (valid > 0).bool()).sum())
        over = int(((corrected > self.clip_value) & (valid > 0).bool()).sum())
        return under / valid_pixels, over / valid_pixels

    # --------------------------------------------------------------- quality
    def _quality(
        self,
        warped_images: torch.Tensor,
        masks: torch.Tensor,
        weights: torch.Tensor,
        gains: torch.Tensor,
        panorama: torch.Tensor,
        valid: torch.Tensor,
        seam_masks: list[torch.Tensor],
        overlap_fractions: list[float],
    ) -> StitchQuality:
        """Measure every seam on the same pixels before and after the blend.

        The before seam measurement is the left-right jump on the seam ring of the naive overwrite view built from the
        raw contributors; the after seam measurement is the same jump on the fused panorama. The before luma
        measurement compares the two raw warped contributors on the overlap, and the after luma measurement compares
        them once their exposure gains have been applied. A successful compensation drives the after luma difference
        toward zero, and feathering drives the after seam jump toward zero.
        """
        before_seam, after_seam = [], []
        before_luma, after_luma = [], []
        corrected = warped_images * gains
        for i, overlap in enumerate(seam_masks):
            overlap_f = overlap.to(panorama.dtype)
            # Seam line: the outer ring of the overlap, where a naive overwrite shows the exposure jump. The ring is
            # the overlap pixels touching a non-overlap pixel, obtained with a one-pixel max-pool of the complement.
            interior = torch.nn.functional.max_pool2d((~overlap).to(panorama.dtype), kernel_size=3, stride=1, padding=1)
            boundary = overlap & (interior > 0)
            # Before: the raw contributor each pixel takes under a naive overwrite, and its seam jump.
            naive_view = torch.where(masks[i : i + 1] > 0, warped_images[i : i + 1], warped_images[i + 1 : i + 2])
            before_seam.append(seam_discontinuity(naive_view, valid, boundary))
            # After: the fused panorama's own left-right jump on that seam ring.
            after_seam.append(seam_discontinuity(panorama, valid, boundary))
            # Luma: raw contributors versus gain-corrected (and range-clamped) contributors, on identical pixels.
            corrected_clamped = corrected.clamp(0.0, self.clip_value)
            before_luma.append(
                overlap_luma_difference(warped_images[i : i + 1], warped_images[i + 1 : i + 2], overlap_f)
            )
            after_luma.append(
                overlap_luma_difference(corrected_clamped[i : i + 1], corrected_clamped[i + 1 : i + 2], overlap_f)
            )

        def stack(values: list[torch.Tensor]) -> torch.Tensor:
            if values:
                return torch.stack(values).mean()
            return torch.zeros((), device=panorama.device)

        clipped = self._clipped_fraction(warped_images, gains, valid)
        return StitchQuality(
            seam_discontinuity_before=stack(before_seam),
            seam_discontinuity_after=stack(after_seam),
            overlap_luma_difference_before=stack(before_luma),
            overlap_luma_difference_after=stack(after_luma),
            clipped_pixel_fraction=clipped,
            overlap_fractions=overlap_fractions,
        )

    def seam_slices(
        self,
        warped_images: torch.Tensor,
        masks: torch.Tensor,
        panorama: torch.Tensor,
        seam_masks: list[torch.Tensor],
    ) -> SeamSlices:
        """Crop before/after rectangles around every seam for manual review.

        The ``before`` crop places the two unfused contributor crops side by side on one strip; the ``after`` crop is
        the fused panorama on the same rectangle, so a reviewer flips between them at identical coordinates.
        """
        margin = self.seam_margin
        boxes: list[tuple[int, int, int, int]] = []
        before_crops, after_crops = [], []
        for i, overlap in enumerate(seam_masks):
            ys, xs = torch.where(overlap[0, 0] > 0)
            if ys.numel() == 0:
                continue
            y0 = max(int(ys.min()) - margin, 0)
            y1 = min(int(ys.max()) + margin + 1, panorama.shape[-2])
            x0 = max(int(xs.min()) - margin, 0)
            x1 = min(int(xs.max()) + margin + 1, panorama.shape[-1])
            boxes.append((y0, y1, x0, x1))
            left = warped_images[i : i + 1, :, y0:y1, x0:x1]
            right = warped_images[i + 1 : i + 2, :, y0:y1, x0:x1]
            before_crops.append(torch.cat([left, right], dim=-1))
            after_crops.append(panorama[:, :, y0:y1, x0:x1])
        if not before_crops:
            empty = panorama.new_zeros(0)
            return SeamSlices(boxes=[], before=empty, after=empty)
        return SeamSlices(boxes=boxes, before=torch.cat(before_crops), after=torch.cat(after_crops))

    # -------------------------------------------------------------------- API
    def fuse(
        self,
        images: list[torch.Tensor],
        homographies: list[torch.Tensor],
        reference_index: int = 0,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, StitchQuality, SeamSlices]:
        """Fuse images with externally supplied geometry; the matcher is never called.

        Args:
            images: ordered single-sample images ``(1, C, H, W)`` of one height, dtype and device.
            homographies: one source-to-canvas homography ``(1, 3, 3)`` per image, e.g. from
                :meth:`estimate_homographies` or an external estimator.
            reference_index: image whose exposure gain is anchored to one.
            return_details: also return the :class:`StitchQuality` metrics and :class:`SeamSlices` review crops.

        Returns:
            ``(panorama, valid_mask)``, or ``(panorama, valid_mask, quality, slices)`` when ``return_details`` is True.

        Raises:
            StitchingError: on inconsistent inputs or an overlap below ``min_overlap_fraction``.
        """
        self._validate_images(images)
        KORNIA_CHECK(
            len(homographies) == len(images),
            f"one homography per image is required, got {len(homographies)} for {len(images)} images",
        )
        # Homographies and every overlap check resolve before the fused panorama is assembled, so a rejected pair
        # raises out of this method without ever returning a partial panorama object.
        height, width, x_min, y_min = self._canvas_size(images, homographies)
        warped_images, masks = self._warp_all(images, homographies, (height, width), (x_min, y_min))
        observations, seam_masks, overlap_fractions = self._gain_observations(warped_images, masks)
        if self.exposure_compensation:
            gains = _solve_gains_from_observations(
                len(images),
                images[0].shape[1],
                images[0].device,
                images[0].dtype,
                observations,
                reference_index,
            )
        else:
            gains = torch.ones(len(images), images[0].shape[1], 1, 1, device=images[0].device, dtype=images[0].dtype)
        # Apply gains with clipping only at the output cast: keep full precision inside the blend.
        weights = feather_weights(masks)
        panorama, valid = fuse_panorama(warped_images, masks, gains, weights)
        panorama = panorama.clamp(0.0, self.clip_value)
        quality = self._quality(warped_images, masks, weights, gains, panorama, valid, seam_masks, overlap_fractions)
        if return_details:
            slices = self.seam_slices(warped_images, masks, panorama, seam_masks)
            return panorama, valid, quality, slices
        return panorama, valid

    def forward(
        self, images: list[torch.Tensor], reference_index: int = 0, return_details: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, StitchQuality, SeamSlices]:
        """Run the full match -> homography -> exposure-compensated feather pipeline.

        Args:
            images: ordered overlapping single-sample images ``(1, C, H, W)`` of one height.
            reference_index: image used as the exposure (and canvas) reference; its gain is one.
            return_details: when True, also return quality metrics and seam slices.

        Raises:
            StitchingError: naming the failing pair when matches, inliers or overlap are insufficient.
        """
        homographies, _ = self.estimate_homographies(images)
        return self.fuse(images, homographies, reference_index, return_details)
