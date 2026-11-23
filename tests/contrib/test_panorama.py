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

from __future__ import annotations

import pytest
import torch
from torch import nn

from kornia.contrib import (
    PanoramaStitcher,
    StitchingError,
    boundary_distance,
    estimate_pair_gain,
    feather_weights,
    fuse_panorama,
    overlap_luma_difference,
    seam_discontinuity,
)
from kornia.feature import LocalFeatureMatcher

from testing.base import BaseTester


def _render(x0: float, w: int, h: int = 48, scale: float = 1.0, base: float = 0.25) -> torch.Tensor:
    """Smooth textured strip whose pixel at canvas column ``x0 + x`` is deterministic, so overlaps align exactly."""
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32) + x0, indexing="ij"
    )
    img = torch.stack(
        [
            base + 0.08 * torch.sin(0.21 * xs + 0.13 * ys + c) + 0.06 * torch.cos(0.17 * ys - 0.07 * xs + 2 * c)
            for c in range(3)
        ]
    )[None]
    return img * scale


class _ShiftMatcher(LocalFeatureMatcher):
    """Exact correspondences of a pure horizontal shift: right pixel p matches left pixel p + (shift, 0)."""

    def __init__(self, shift: float, keep: int | None = None) -> None:
        nn.Module.__init__(self)
        self.shift, self.keep = shift, keep

    def forward(self, data: dict) -> dict:
        image = data["image1"]
        ys, xs = torch.meshgrid(
            torch.arange(2, image.shape[-2] - 2, 5.0), torch.arange(2, image.shape[-1] - 2, 5.0), indexing="ij"
        )
        kp1 = torch.stack([xs.flatten(), ys.flatten()], -1)
        kp0 = kp1 + torch.tensor([self.shift, 0.0])
        idx = torch.zeros(kp0.shape[0], dtype=torch.long)
        if self.keep is not None:
            kp0, kp1, idx = kp0[: self.keep], kp1[: self.keep], idx[: self.keep]
        return {"keypoints0": kp0, "keypoints1": kp1, "batch_indexes": idx}


class _BoomMatcher(LocalFeatureMatcher):
    """A matcher that must never run; used to prove fusion-only mode ignores the matcher."""

    def __init__(self) -> None:
        nn.Module.__init__(self)

    def forward(self, data: dict) -> dict:
        raise AssertionError("the matcher must not be called in fusion-only mode")


def _shift_homography(shift_x: float) -> torch.Tensor:
    return (torch.eye(3) + torch.tensor([[0.0, 0.0, shift_x], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))[None]


class TestPanoramaHelpers(BaseTester):
    def test_boundary_distance_grows_inward(self, device, dtype):
        mask = torch.zeros(1, 1, 8, 8, device=device, dtype=dtype)
        mask[:, :, 2:6, 2:6] = 1.0
        dist = boundary_distance(mask)
        # A covered pixel next to the boundary is one pixel from the nearest uncovered pixel; the centre of the 4x4
        # interior is two pixels from every edge.
        self.assert_close(dist[0, 0, 4, 4].float(), torch.tensor(2.0), atol=1e-5, rtol=0)
        assert float(dist[0, 0, 2, 2]) == 1.0
        assert dist.shape == mask.shape

    def test_boundary_distance_empty_and_full(self, device, dtype):
        empty = torch.zeros(1, 1, 6, 6, device=device, dtype=dtype)
        full = torch.ones(1, 1, 6, 6, device=device, dtype=dtype)
        assert float(boundary_distance(empty).abs().sum()) == 0.0
        assert float(boundary_distance(full).abs().sum()) == 0.0  # no interior pixel

    def test_feather_weights_normalise_and_favour_interior(self, device, dtype):
        masks = torch.zeros(2, 1, 4, 8, device=device, dtype=dtype)
        masks[0, :, :, :5] = 1.0  # left image covers columns 0..4
        masks[1, :, :, 3:] = 1.0  # right image covers columns 3..7; overlap at 3..4
        weights = feather_weights(masks)
        total = weights.sum(0)
        covered = masks.sum(0) > 0
        self.assert_close(total[covered], torch.ones_like(total[covered]), atol=1e-5, rtol=0)
        # A pixel covered by only one image takes all of the weight.
        assert float(weights[0, 0, 0, 0]) == 1.0
        assert float(weights[1, 0, 0, 7]) == 1.0
        # The weight is explainable: it equals the ratio of the two boundary distances.
        dist = boundary_distance(masks)
        expected = dist[1] / dist.sum(0).clamp_min(torch.finfo(dist.dtype).tiny)
        self.assert_close(weights[1], expected, atol=1e-5, rtol=0)

    def test_estimate_pair_gain_recovers_known_multiplier(self, device, dtype):
        torch.manual_seed(0)
        base = torch.rand(1, 3, 12, 16, device=device).clamp(0.1, 0.9).to(dtype)
        true_gain = 1.7
        shifted = base / true_gain  # shifted * true_gain == base
        overlap = torch.ones(1, 1, 12, 16, device=device, dtype=dtype)
        gain = estimate_pair_gain(base, shifted, overlap)
        tol = 2e-2 if dtype == torch.bfloat16 else 5e-3
        self.assert_close(gain.float().flatten(), torch.full((3,), true_gain), atol=tol, rtol=tol)

    def test_estimate_pair_gain_empty_overlap_is_identity(self, device, dtype):
        a = torch.rand(1, 3, 6, 6, device=device, dtype=dtype)
        b = torch.rand(1, 3, 6, 6, device=device, dtype=dtype)
        gain = estimate_pair_gain(a, b, torch.zeros_like(a[:, :1]))
        self.assert_close(gain, torch.ones_like(gain))

    def test_fuse_panorama_uses_weights_and_mask(self, device, dtype):
        # Two constant contributors: the blended overlap must be the distance-weighted mean, not an overwrite.
        images = torch.zeros(2, 3, 4, 6, device=device, dtype=dtype)
        images[0, :, :, :4] = 0.8
        images[1, :, :, 2:] = 0.2
        masks = torch.zeros(2, 1, 4, 6, device=device, dtype=dtype)
        masks[0, :, :, :4] = 1.0
        masks[1, :, :, 2:] = 1.0
        pano, valid = fuse_panorama(images, masks)
        assert pano.shape == (1, 3, 4, 6)
        # Single-coverage regions keep their only contributor's value.
        self.assert_close(pano[0, :, 0, 0], images[0, :, 0, 0])
        self.assert_close(pano[0, :, 0, 5], images[1, :, 0, 5])
        # Every overlap pixel is strictly between the two values: no wholesale overwrite.
        overlap_cols = pano[0, :, :, 2:4]
        assert bool((overlap_cols > 0.2 + 1e-4).all() and (overlap_cols < 0.8 - 1e-4).all())
        assert bool((valid > 0).all())

    def test_seam_discontinuity_zero_on_flat_image(self, device, dtype):
        image = torch.full((1, 3, 4, 8), 0.5, device=device, dtype=dtype)
        mask = torch.ones(1, 1, 4, 8, device=device, dtype=dtype)
        seam = torch.zeros(1, 1, 4, 8, device=device, dtype=dtype)
        seam[:, :, :, 3:5] = 1.0
        assert float(seam_discontinuity(image, mask, seam)) == 0.0

    def test_overlap_luma_difference_matches_manual_mean(self, device, dtype):
        a = torch.full((1, 3, 2, 4), 0.6, device=device, dtype=dtype)
        b = torch.full((1, 3, 2, 4), 0.2, device=device, dtype=dtype)
        overlap = torch.ones(1, 1, 2, 4, device=device, dtype=dtype)
        tol = 1e-2 if dtype == torch.bfloat16 else 5e-3
        self.assert_close(overlap_luma_difference(a, b, overlap).float(), torch.tensor(0.4), atol=tol, rtol=0)

    def test_gradcheck_estimate_pair_gain(self, device):
        a = torch.rand(1, 3, 6, 6, device=device, dtype=torch.float64, requires_grad=True)
        b = torch.rand(1, 3, 6, 6, device=device, dtype=torch.float64, requires_grad=True)
        overlap = torch.ones(1, 1, 6, 6, device=device, dtype=torch.float64)
        self.gradcheck(lambda x, y: estimate_pair_gain(x, y, overlap), (a, b))

    def test_fuse_panorama_passes_gradients(self, device):
        # The full feather blend is differentiable: every contributor receives a finite, non-zero gradient.
        masks = torch.zeros(2, 1, 8, 12, device=device)
        masks[0, :, :, :8] = 1.0
        masks[1, :, :, 4:] = 1.0
        images = torch.rand(2, 3, 8, 12, device=device, dtype=torch.float64, requires_grad=True)
        panorama, _ = fuse_panorama(images, masks)
        panorama.sum().backward()
        assert images.grad is not None
        assert bool(torch.isfinite(images.grad).all())
        # An overlap pixel is touched by both contributors, so both gradients are non-zero there.
        assert float(images.grad[:, :, :, 4:8].abs().sum()) > 0.0

    def test_dynamo(self, device, dtype, torch_optimizer):
        masks = (torch.rand(2, 1, 12, 16, device=device) > 0.2).to(dtype)
        masks[:, :, :, :2] = 1.0
        images = torch.rand(2, 3, 12, 16, device=device, dtype=dtype) * masks
        weights = torch_optimizer(feather_weights)(masks)
        self.assert_close(weights, feather_weights(masks), atol=1e-5, rtol=1e-5)
        pano, valid = torch_optimizer(fuse_panorama)(images, masks)
        pano_ref, valid_ref = fuse_panorama(images, masks)
        self.assert_close(pano, pano_ref, atol=1e-5, rtol=1e-5)
        self.assert_close(valid, valid_ref, atol=1e-5, rtol=1e-5)


class TestPanoramaStitcher(BaseTester):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.left = _render(0.0, 64)
        # Brightness and white-balance difference: a per-channel scale, kept below clipping for the metric tests.
        self.right = _render(40.0, 56, scale=1.5)
        self.shift = 40.0

    def test_smoke(self, device, dtype):
        stitcher = PanoramaStitcher(_ShiftMatcher(self.shift)).to(device=device, dtype=dtype)
        pano, mask = stitcher([self.left.to(device, dtype), self.right.to(device, dtype)])
        assert pano.shape[:3] == (1, 3, 48)
        assert pano.shape[-1] == 96
        assert pano.dtype == dtype
        assert mask.shape == (1, 1, 48, pano.shape[-1])
        assert bool(torch.isfinite(pano).all())

    def test_seam_discontinuity_drops_with_numbers(self, device):
        # Acceptance 1: the seam discontinuity must fall, and the test pins the actual before/after numbers.
        stitcher = PanoramaStitcher(_ShiftMatcher(self.shift))
        _, _, quality, _ = stitcher([self.left, self.right], return_details=True)
        before = float(quality.seam_discontinuity_before)
        after = float(quality.seam_discontinuity_after)
        assert after < before * 0.4  # at least a 60% reduction
        # Pinned measurements on this deterministic scene (regression guard; regenerate only with the scene fixed).
        assert before == pytest.approx(0.0678, abs=5e-3)
        assert after == pytest.approx(0.0109, abs=5e-3)

    def test_same_material_color_difference_within_tolerance(self, device):
        # Acceptance 2: after compensation, the same material on both views agrees in the overlap.
        stitcher = PanoramaStitcher(_ShiftMatcher(self.shift))
        _, _, quality, _ = stitcher([self.left, self.right], return_details=True)
        assert float(quality.overlap_luma_difference_after) < 0.01
        assert float(quality.overlap_luma_difference_after) < float(quality.overlap_luma_difference_before) * 0.05

    def test_matching_failure_is_rejected_with_pair_and_counts(self, device):
        # Acceptance 3: too few inliers names the pair and how many inliers were found.
        stitcher = PanoramaStitcher(_ShiftMatcher(self.shift, keep=3), min_inliers=8)
        with pytest.raises(StitchingError, match=r"pair \(0, 1\).*3 matched keypoints"):
            stitcher([self.left, self.right])

    def test_small_overlap_is_rejected(self, device):
        identity = torch.eye(3)[None]
        disjoint = _shift_homography(200.0)
        stitcher = PanoramaStitcher(_BoomMatcher())
        with pytest.raises(StitchingError, match=r"pair \(0, 1\).*overlap is 0\.0000"):
            stitcher.fuse([self.left, self.right], [identity, disjoint])

    def test_two_runs_are_bit_identical(self, device):
        # Acceptance 4: same inputs, fresh stitcher instances (seeded RANSAC), bit-for-bit identical output.
        images = [self.left, self.right]
        first = PanoramaStitcher(_ShiftMatcher(self.shift))
        second = PanoramaStitcher(_ShiftMatcher(self.shift))
        pano_a, mask_a = first(images)
        pano_b, mask_b = second(images)
        assert torch.equal(pano_a, pano_b)
        assert torch.equal(mask_a, mask_b)

    def test_fusion_only_never_calls_the_matcher(self, device):
        # Requirement 8: externally supplied geometry goes straight to fusion, matcher untouched.
        stitcher = PanoramaStitcher(_BoomMatcher())
        homographies = [torch.eye(3)[None], _shift_homography(self.shift)]
        pano, mask = stitcher.fuse([self.left, self.right], homographies)
        assert pano.shape[-1] == 96
        assert bool(mask.bool().any())

    def test_three_image_chain_does_not_diverge(self, device):
        # Requirement 3: a chain of three images keeps the same per-seam quality order as a single pair.
        images = [_render(0.0, 64), _render(40.0, 64, scale=1.3), _render(80.0, 64, scale=1.3 * 1.3)]
        stitcher = PanoramaStitcher(_ShiftMatcher(40.0))
        pano, _mask, quality, _ = stitcher(images, return_details=True)
        assert pano.shape[-1] == 144
        assert bool(torch.isfinite(pano).all())
        assert float(quality.overlap_luma_difference_after) < 0.01
        assert float(quality.seam_discontinuity_after) < 0.03
        assert len(quality.overlap_fractions) == 2

    def test_reference_frame_anchors_its_gain(self, device):
        # Requirement 4: the reference image's gain is exactly one and the other image moves to it.
        images = [self.left, self.right]
        stitcher = PanoramaStitcher(_ShiftMatcher(self.shift))
        homographies, _ = stitcher.estimate_homographies(images)
        height, width, x_min, y_min = stitcher._canvas_size(images, homographies)
        warped, masks = stitcher._warp_all(images, homographies, (height, width), (x_min, y_min))
        observations, _, _ = stitcher._gain_observations(warped, masks)
        from kornia.contrib.panorama import _solve_gains_from_observations

        gains = _solve_gains_from_observations(2, 3, device, torch.float32, observations, reference_index=0)
        assert gains.shape == (2, 3, 1, 1)
        self.assert_close(gains[0, :, 0, 0], torch.ones(3, device=device), atol=1e-6, rtol=0)
        # The right image is 1.5x brighter, so its compensating gain is around 1/1.5.
        assert float(gains[1].mean()) == pytest.approx(1.0 / 1.5, abs=0.02)

    def test_clipped_pixels_are_flagged(self, device):
        # Requirement (failure 2): a dark right image whose fitted gain lifts its non-overlap content past the value
        # range gets a non-zero over-exposure fraction in the quality report.
        dark = _render(40.0, 56, scale=0.4)
        # The non-overlapping right strip carries a bright band the overlap statistics never see, so the overlap-only
        # gain (~2.5) pushes it above the clip value after compensation.
        dark[:, :, :, 48:] = 0.9
        stitcher = PanoramaStitcher(_ShiftMatcher(self.shift))
        _, _, quality, _ = stitcher([self.left, dark], return_details=True)
        under, over = quality.clipped_pixel_fraction
        assert over > 0.0
        assert under == 0.0

    def test_inconsistent_inputs_are_rejected(self, device):
        stitcher = PanoramaStitcher(_ShiftMatcher(self.shift))
        with pytest.raises(StitchingError, match="channel count"):
            stitcher([self.left, torch.rand(1, 4, 48, 56)])
        with pytest.raises(StitchingError, match="one height"):
            stitcher([self.left, torch.rand(1, 3, 40, 56)])
        with pytest.raises(StitchingError, match="device/dtype"):
            stitcher([self.left, self.right.double()])
        with pytest.raises(Exception):  # ShapeError from KORNIA_CHECK_SHAPE
            stitcher([self.left, torch.rand(3, 48, 56)])
        with pytest.raises(Exception, match="at least two images"):  # BaseError from KORNIA_CHECK
            stitcher([self.left])

    def test_failure_leaves_no_panorama(self, device):
        # Requirement (failure 4): a rejected pair raises before any panorama object is returned.
        stitcher = PanoramaStitcher(_ShiftMatcher(self.shift, keep=2))
        with pytest.raises(StitchingError):
            stitcher([self.left, self.right])

    def test_order_swap_has_comparable_seam_bound(self, device):
        # Requirement 11: swapping the order changes the seam metric by a small bounded amount, not unbounded.
        stitcher = PanoramaStitcher(_ShiftMatcher(self.shift))
        _, _, q_ab, _ = stitcher([self.left, self.right], return_details=True)
        inverse = torch.linalg.inv(_shift_homography(self.shift)[0])[None]
        _, _, q_ba, _ = stitcher.fuse([self.right, self.left], [torch.eye(3)[None], inverse], return_details=True)
        delta = abs(float(q_ab.seam_discontinuity_after) - float(q_ba.seam_discontinuity_after))
        assert delta < 0.02  # both orderings land within two hundredths on this smooth scene

    def test_seam_slices_pair_before_and_after(self, device):
        # Requirement 12: every seam yields aligned before/after review crops.
        stitcher = PanoramaStitcher(_ShiftMatcher(self.shift))
        homographies, _ = stitcher.estimate_homographies([self.left, self.right])
        height, width, x_min, y_min = stitcher._canvas_size([self.left, self.right], homographies)
        warped, masks = stitcher._warp_all([self.left, self.right], homographies, (height, width), (x_min, y_min))
        pano, _, _, _ = stitcher.fuse([self.left, self.right], homographies, return_details=True)
        _, seams, _ = stitcher._gain_observations(warped, masks)
        slices = stitcher.seam_slices(warped, masks, pano, seams)
        assert len(slices.boxes) == 1
        assert slices.before.shape[0] == slices.after.shape[0] == 1
        # The "before" strip holds both contributor crops side by side, so it is twice as wide.
        assert slices.before.shape[-1] == 2 * slices.after.shape[-1]

    def test_disabling_compensation_keeps_gains_identity(self, device):
        stitcher = PanoramaStitcher(_ShiftMatcher(self.shift), exposure_compensation=False)
        pano, _, quality, _ = stitcher([self.left, self.right], return_details=True)
        assert bool(torch.isfinite(pano).all())
        # Without compensation the raw exposure difference survives, demonstrating the flag's effect.
        assert float(quality.overlap_luma_difference_after) == pytest.approx(
            float(quality.overlap_luma_difference_before), rel=0.05
        )

    @pytest.mark.parametrize("img_dtype", [torch.float16, torch.bfloat16])
    def test_half_precision(self, img_dtype, device):
        stitcher = PanoramaStitcher(_ShiftMatcher(self.shift))
        pano, mask, quality, _ = stitcher(
            [self.left.to(device, img_dtype), self.right.to(device, img_dtype)], return_details=True
        )
        assert pano.dtype == img_dtype
        assert mask.dtype == img_dtype
        assert bool(torch.isfinite(pano.float()).all())
        assert float(quality.seam_discontinuity_after) < 0.03

    def test_naive_overwrite_is_not_produced(self, device):
        # Hard constraint 2: the later image must not replace the overlap wholesale.
        stitcher = PanoramaStitcher(_ShiftMatcher(self.shift))
        homographies, _ = stitcher.estimate_homographies([self.left, self.right])
        pano, _ = stitcher.fuse([self.left, self.right], homographies)
        height, width, x_min, y_min = stitcher._canvas_size([self.left, self.right], homographies)
        warped, masks = stitcher._warp_all([self.left, self.right], homographies, (height, width), (x_min, y_min))
        overlap = ((masks[0:1] > 0) & (masks[1:2] > 0)).bool()
        # On overlap pixels the fused result differs from the raw right image wherever the two disagree.
        diff_right = (pano - warped[1:2]).abs().sum(1, keepdim=True)
        assert float(diff_right[overlap].mean()) > 1e-3
