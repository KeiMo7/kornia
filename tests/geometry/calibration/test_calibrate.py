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

"""Tests for the multi-view camera calibration entry points."""

from typing import List, Tuple

import torch

from kornia.geometry.calibration import (
    PINHOLE_MODEL,
    CalibrationObservations,
    calibrate_camera,
    calibrate_cameras,
    calibrate_fisheye,
    distort_points,
)
from kornia.geometry.camera import distort_points_kannala_brandt

from testing.base import BaseTester

# Hardcoded ground-truth camera used to synthesize every planar scene.
_TRUE_K = torch.tensor(
    [[700.0, 0.0, 320.0], [0.0, 705.0, 240.0], [0.0, 0.0, 1.0]], dtype=torch.float64
)
_TRUE_DIST = torch.tensor([-0.28, 0.09, 0.001, -0.002, 0.03], dtype=torch.float64)
_IMAGE_SIZE = (480, 640)


def _planar_board(cols: int = 9, rows: int = 7, sx: float = 0.03, sy: float = 0.035) -> torch.Tensor:
    gx = (torch.arange(cols, dtype=torch.float64) - (cols - 1) / 2) * sx
    gy = (torch.arange(rows, dtype=torch.float64) - (rows - 1) / 2) * sy
    bx, by = torch.meshgrid(gx, gy, indexing="xy")
    return torch.stack(
        [bx.reshape(-1), by.reshape(-1), torch.zeros(cols * rows, dtype=torch.float64)], dim=-1
    )


def _view_pose(view: int, n_views: int, generator: torch.Generator) -> torch.Tensor:
    """Deterministic, varied but in-frame world-to-camera pose, with a small seeded jitter."""
    angle = 2 * torch.pi * view / max(n_views, 1)
    # A moderate tilt keeps the board spanning both image axes; distance varies to add scale diversity.
    tilt = 0.15 + 0.12 * torch.rand((), generator=generator, dtype=torch.float64)
    distance = 1.3 + 0.25 * torch.rand((), generator=generator, dtype=torch.float64)
    cam_pos = torch.stack(
        [
            distance * tilt * torch.cos(torch.tensor(angle, dtype=torch.float64)),
            distance * tilt * torch.sin(torch.tensor(angle, dtype=torch.float64)),
            distance,
        ]
    )
    view_dir = -cam_pos / cam_pos.norm()
    ref_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    right = torch.linalg.cross(view_dir, ref_up)
    right = right / right.norm()
    up = torch.linalg.cross(right, view_dir)
    # A small, varying roll so the board orientation changes between views.
    roll = 0.2 * torch.sin(torch.tensor(2.0 * angle, dtype=torch.float64))
    right = right * torch.cos(roll) + up * torch.sin(roll)
    up = torch.linalg.cross(right, view_dir)
    r_cw = torch.stack([right, up, view_dir], dim=0)
    r_wc = r_cw.transpose(-1, -2).contiguous()
    t_wc = -r_wc @ cam_pos
    return torch.cat([r_wc, t_wc[:, None]], dim=-1)


def _planar_scene(
    n_views: int,
    noise_px: float,
    seed: int,
    dist: torch.Tensor = _TRUE_DIST,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Synthesize planar-board views with varied, in-frame camera poses."""
    board = _planar_board()
    generator = torch.Generator().manual_seed(seed)
    world_pts, img_pts = [], []
    for view in range(n_views):
        rt = _view_pose(view, n_views, generator)
        cam = torch.cat([board, torch.ones(board.shape[0], 1, dtype=torch.float64)], dim=-1) @ rt.T
        assert bool((cam[:, 2] > 0.05).all())
        ideal = cam @ _TRUE_K.T
        ideal = ideal[:, :2] / ideal[:, 2:3]
        measured = distort_points(ideal, _TRUE_K, dist)
        if noise_px > 0:
            measured = (
                measured
                + torch.randn(measured.shape, generator=generator, dtype=torch.float64) * noise_px
            )
        world_pts.append(board.clone())
        img_pts.append(measured)
    return world_pts, img_pts


class TestCalibrateCamera(BaseTester):
    def test_noiseless_recovers_ground_truth(self, device):
        world_pts, img_pts = _planar_scene(14, noise_px=0.0, seed=1)
        world_pts = [w.to(device) for w in world_pts]
        img_pts = [i.to(device) for i in img_pts]
        result = calibrate_camera(world_pts, img_pts, image_size=_IMAGE_SIZE, camera_id="cam", distortion_order=5)
        assert result.success
        assert result.model == PINHOLE_MODEL
        assert result.image_size == _IMAGE_SIZE
        assert result.camera_id == "cam"
        # Noiseless data must recover the intrinsics and distortion to high precision.
        self.assert_close(result.intrinsics, _TRUE_K.to(device), rtol=1e-6, atol=1e-4)
        self.assert_close(result.distortion, _TRUE_DIST.to(device), rtol=1e-6, atol=1e-6)
        assert result.rms_reprojection_error < 1e-6
        assert result.extrinsics.shape == (14, 3, 4)

    def test_noisy_recovery_and_residual_consistency(self, device):
        world_pts, img_pts = _planar_scene(14, noise_px=0.3, seed=2)
        world_pts = [w.to(device) for w in world_pts]
        img_pts = [i.to(device) for i in img_pts]
        result = calibrate_camera(world_pts, img_pts, image_size=_IMAGE_SIZE, distortion_order=5)
        assert result.success
        # The RMS must be of the same order as the injected pixel noise.
        assert result.rms_reprojection_error < 1.0
        # Focal lengths within a few percent of the truth.
        assert abs(result.intrinsics[0, 0].item() - 700.0) / 700.0 < 0.05
        assert abs(result.intrinsics[1, 1].item() - 705.0) / 705.0 < 0.05
        # An independent re-projection through the public API must match the reported RMS.
        total_sq, total_n = 0.0, 0
        for local, diag in enumerate(d for d in result.view_diagnostics if d.used):
            view_index = diag.view_index
            cam = (
                torch.cat(
                    [world_pts[view_index], torch.ones(world_pts[view_index].shape[0], 1, device=device)], dim=-1
                )
                @ result.extrinsics[local].T
            )
            ideal = cam @ result.intrinsics.T
            ideal = ideal[:, :2] / ideal[:, 2:3]
            projected = distort_points(ideal, result.intrinsics, result.distortion)
            err = torch.linalg.vector_norm(projected - img_pts[view_index], dim=-1)
            mask = result.inlier_masks[view_index]
            total_sq += float(err[mask].square().sum())
            total_n += int(mask.sum())
        independent_rms = (total_sq / total_n) ** 0.5
        self.assert_close(independent_rms, result.rms_reprojection_error, rtol=1e-9, atol=1e-9)

    def test_gross_point_outliers_are_rejected(self, device):
        world_pts, img_pts = _planar_scene(14, noise_px=0.3, seed=3)
        generator = torch.Generator().manual_seed(7)
        injected = 0
        for v in range(0, 14, 2):
            idx = torch.randint(0, img_pts[v].shape[0], (6,), generator=generator)
            img_pts[v][idx] += torch.randn((6, 2), generator=generator, dtype=torch.float64) * 60
            injected += 6
        result = calibrate_camera(world_pts, img_pts, image_size=_IMAGE_SIZE, distortion_order=5)
        assert result.success
        # Most of the injected gross errors must be flagged; without robust rejection the focal length diverges by
        # tens of percent, while here it stays within ten percent of the truth.
        assert sum(result.rejected_points) >= 0.8 * injected
        assert abs(result.intrinsics[0, 0].item() - 700.0) / 700.0 < 0.10
        assert result.rms_reprojection_error < 1.0

    def test_junk_view_is_reported_and_dropped(self, device):
        world_pts, img_pts = _planar_scene(14, noise_px=0.3, seed=4)
        perm = torch.randperm(img_pts[5].shape[0], generator=torch.Generator().manual_seed(3))
        img_pts[5] = img_pts[5][perm] + 80
        result = calibrate_camera(world_pts, img_pts, image_size=_IMAGE_SIZE, distortion_order=5)
        assert result.success
        assert 5 in result.rejected_views
        junk = next(d for d in result.view_diagnostics if d.view_index == 5)
        assert junk.rejected
        assert junk.rejection_reason is not None
        # The remaining views still calibrate accurately.
        assert abs(result.intrinsics[0, 0].item() - 700.0) / 700.0 < 0.05

    def test_degenerate_inputs_fail_with_reasons(self, device):
        world_pts, img_pts = _planar_scene(6, noise_px=0.1, seed=5)
        no_views = calibrate_camera([], [], image_size=_IMAGE_SIZE)
        assert not no_views.success
        assert no_views.rejection_reasons

        few_points = calibrate_camera(
            [world_pts[0][:3]], [img_pts[0][:3]], image_size=_IMAGE_SIZE
        )
        assert not few_points.success
        assert few_points.view_diagnostics[0].degeneracy == "fewer_than_four_points"

        zero_weights = [torch.zeros(w.shape[0], dtype=torch.float64) for w in world_pts]
        weighted = calibrate_camera(world_pts, img_pts, weights=zero_weights, image_size=_IMAGE_SIZE)
        assert not weighted.success
        assert weighted.rejection_reasons

        nan_pts = [torch.full_like(img_pts[0], float("nan"))]
        non_finite = calibrate_camera([world_pts[0]], nan_pts, image_size=_IMAGE_SIZE)
        assert not non_finite.success

    def test_deterministic_repeat(self, device):
        world_pts, img_pts = _planar_scene(12, noise_px=0.3, seed=6)
        first = calibrate_camera(world_pts, img_pts, image_size=_IMAGE_SIZE, distortion_order=5)
        second = calibrate_camera(world_pts, img_pts, image_size=_IMAGE_SIZE, distortion_order=5)
        assert torch.equal(first.intrinsics, second.intrinsics)
        assert torch.equal(first.distortion, second.distortion)
        assert torch.equal(first.extrinsics, second.extrinsics)

    def test_half_precision_rounded_to_input_dtype(self, device):
        world_pts, img_pts = _planar_scene(10, noise_px=0.3, seed=8)
        world_pts = [w.to(torch.float16) for w in world_pts]
        img_pts = [i.to(torch.float16) for i in img_pts]
        result = calibrate_camera(world_pts, img_pts, image_size=_IMAGE_SIZE, distortion_order=5)
        assert result.success
        assert result.intrinsics.dtype == torch.float16
        assert result.distortion.dtype == torch.float16


class TestCalibrateCamerasBatch(BaseTester):
    def test_batch_isolation_and_parallel_equivalence(self, device):
        good_world, good_img = _planar_scene(10, noise_px=0.3, seed=9)
        good = CalibrationObservations(good_world, good_img, image_size=_IMAGE_SIZE, camera_id="good")
        # A single view cannot calibrate and must fail on its own.
        bad = CalibrationObservations(good_world[:1], good_img[:1], image_size=_IMAGE_SIZE, camera_id="bad")
        serial = calibrate_cameras([good, bad], model=PINHOLE_MODEL, distortion_order=5)
        parallel = calibrate_cameras(
            [good, bad], model=PINHOLE_MODEL, distortion_order=5, parallel=True
        )
        assert [r.success for r in serial] == [True, False]
        assert [r.camera_id for r in serial] == ["good", "bad"]
        assert serial[1].rejection_reasons
        # Parallel and serial runs must be bit-identical and the failed camera cannot pollute the good one.
        assert torch.equal(serial[0].intrinsics, parallel[0].intrinsics)
        assert torch.equal(serial[0].distortion, parallel[0].distortion)


def _fisheye_scene(n_views: int, noise_px: float, seed: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Synthesize planar-board views through the Kannala-Brandt model, using deterministic poses."""
    board = _planar_board()
    true_flat = torch.tensor(
        [700.0, 705.0, 320.0, 240.0, 0.08, -0.02, 0.005, -0.001], dtype=torch.float64
    )
    generator = torch.Generator().manual_seed(seed)
    world_pts, img_pts = [], []
    for view in range(n_views):
        rt = _view_pose(view, n_views, generator)
        cam = torch.cat([board, torch.ones(board.shape[0], 1, dtype=torch.float64)], dim=-1) @ rt.T
        assert bool((cam[:, 2] > 0.05).all())
        normalized = cam[:, :2] / cam[:, 2:3]
        measured = distort_points_kannala_brandt(normalized, true_flat)
        if noise_px > 0:
            measured = measured + torch.randn(measured.shape, generator=generator, dtype=torch.float64) * noise_px
        world_pts.append(board.clone())
        img_pts.append(measured)
    return world_pts, img_pts


class TestCalibrateFisheye(BaseTester):
    _TRUE_FLAT = torch.tensor(
        [700.0, 705.0, 320.0, 240.0, 0.08, -0.02, 0.005, -0.001], dtype=torch.float64
    )

    def test_noiseless_fisheye_recovers_flat_params(self, device):
        world_pts, img_pts = _fisheye_scene(12, noise_px=0.0, seed=11)
        result = calibrate_fisheye(world_pts, img_pts, image_size=_IMAGE_SIZE, camera_id="fish")
        assert result.success
        assert result.distortion.shape == (8,)
        self.assert_close(result.distortion, self._TRUE_FLAT, rtol=1e-6, atol=1e-4)
        assert result.rms_reprojection_error < 1e-6

    def test_noisy_fisheye_residual_level(self, device):
        world_pts, img_pts = _fisheye_scene(12, noise_px=0.2, seed=12)
        result = calibrate_fisheye(world_pts, img_pts, image_size=_IMAGE_SIZE)
        assert result.success
        assert result.rms_reprojection_error < 1.0
        # The intrinsics embedded in the flat vector stay accurate.
        assert abs(result.distortion[0].item() - 700.0) / 700.0 < 0.05
