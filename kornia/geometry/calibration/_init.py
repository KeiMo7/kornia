# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
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

"""Initial camera estimates for calibration: planar Zhang homographies, the non-planar PnP fallback and checks.

A planar calibration board gives, per view, a homography from board-plane coordinates to measured pixels; Zhang's
closed-form extraction turns the homographies of several views into one pinhole matrix and one pose per view.
Non-planar world points fall back to a nominal pinhole matrix and the DLT PnP pose of every view. Every routine
here reports a named degeneracy instead of letting an underlying linear algebra failure propagate.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from kornia.core.utils import _at_least_float32
from kornia.geometry.calibration.pnp import solve_pnp_dlt
from kornia.geometry.conversions import rotation_matrix_to_axis_angle
from kornia.geometry.homography import find_homography_dlt

#: A view needs at least this many points for a homography or a meaningful pose.
MIN_POINTS_PER_VIEW = 4
#: Fraction of the image a view's points must span (relative extent) to count as covering the frame.
MIN_SPREAD_RATIO = 0.02
#: Relative camera-to-board translation change below which a view is considered stationary against the median.
MIN_BASELINE_RATIO = 0.05
#: Rank-one singular-value ratio below which a board is treated as a line rather than a plane.
PLANARITY_RANK_EPS = 1e-6
#: Minimum fraction of a view's points that the robust homography must keep for the view to enter the fit.
_MIN_INLIER_FRACTION = 0.25


@dataclass
class InitialEstimate:
    r"""Initial parameters and per-view flags produced before nonlinear refinement.

    Attributes:
        intrinsic_params: Initial ``(fx, fy, cx, cy)`` vector.
        pose_params: Initial per-view ``(axis-angle, translation)`` vectors, shape ``(M, 6)``.
        view_valid: Bool tensor ``(M,)`` of views that passed the degeneracy checks.
        view_reasons: Per-view rejection reason strings, None for accepted views.
        is_planar: Whether the world points were treated as a planar board.
        fatal_reason: A reason that makes the whole calibration impossible, or None.
    """

    intrinsic_params: Optional[torch.Tensor]
    pose_params: Optional[torch.Tensor]
    view_valid: torch.Tensor
    view_reasons: List[Optional[str]]
    is_planar: bool
    fatal_reason: Optional[str] = None
    inlier_masks: Optional[List[Optional[torch.Tensor]]] = None


def _world_planarity(world_points: Sequence[torch.Tensor], eps: float = PLANARITY_RANK_EPS) -> bool:
    """Whether every view's world points lie in one plane (all views share one board in the common case)."""
    for points in world_points:
        centered = points - points.mean(dim=0, keepdim=True)
        singular = torch.linalg.svdvals(centered)
        scale = singular[0].clamp_min(eps)
        if bool(singular[-1] > eps * scale):
            return False
    return True


def _point_spread_ok(image_points: torch.Tensor, image_size: Optional[Tuple[int, int]]) -> bool:
    extent = image_points.amax(dim=0) - image_points.amin(dim=0)
    if image_size is not None:
        height, width = image_size
        return bool(extent[0] >= MIN_SPREAD_RATIO * width and extent[1] >= MIN_SPREAD_RATIO * height)
    # Without a declared resolution, require the points not to collapse onto a line or a point.
    centered = image_points - image_points.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    return bool(singular[-1] > PLANARITY_RANK_EPS * singular[0])


def _planar_xy(world_points: torch.Tensor) -> torch.Tensor:
    """Project planar world points onto their two spanning plane axes as board coordinates."""
    centered = world_points - world_points.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:2].transpose(-1, -2)


def _homography_view(
    board_xy: torch.Tensor, image_points: torch.Tensor, weights: Optional[torch.Tensor]
) -> torch.Tensor:
    return find_homography_dlt(
        board_xy[None], image_points[None], None if weights is None else weights[None]
    )[0]


def _robust_homography_view(
    board_xy: torch.Tensor, image_points: torch.Tensor, inlier_threshold: float
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """RANSAC homography of one planar view, returning the model and its inlier mask.

    A fixed seed makes the run deterministic and independent of any global random state. Returns ``(None, None)``
    when RANSAC cannot find a model with the minimum four-point support.
    """
    from kornia.geometry.ransac import RANSAC

    estimator = RANSAC(
        model_type="homography",
        inl_th=inlier_threshold,
        max_iter=100,
        confidence=0.999,
        seed=0,
    )
    model, inliers = estimator(board_xy, image_points)
    if not bool(inliers.any()) or not bool(torch.isfinite(model).all()):
        return None, None
    # Refit on the RANSAC inliers with the normalized DLT for a sub-pixel homography.
    h = find_homography_dlt(board_xy[None], image_points[None], inliers.to(board_xy.dtype)[None])[0]
    if not bool(torch.isfinite(h).all()):
        return None, None
    return h, inliers


def _intrinsics_from_homographies(homographies: torch.Tensor) -> Optional[torch.Tensor]:
    r"""Zhang's closed-form intrinsic extraction from view homographies, None when the equations are degenerate."""
    # V rows of Zhang's constraints on b = (B11, B12, B22, B13, B23, B33), B = K^-T K^-1.
    rows = []
    for h in homographies:
        h1 = h[:, 0]
        h2 = h[:, 1]
        def v_ij(hi, hj):
            return torch.stack(
                    [
                        hi[0] * hj[0],
                        hi[0] * hj[1] + hi[1] * hj[0],
                        hi[1] * hj[1],
                        hi[2] * hj[0] + hi[0] * hj[2],
                        hi[2] * hj[1] + hi[1] * hj[2],
                        hi[2] * hj[2],
                    ]
                )
        rows.append(v_ij(h1, h2))
        rows.append(v_ij(h1, h1) - v_ij(h2, h2))
    v = torch.stack(rows, dim=0)
    if not bool(torch.isfinite(v).all()):
        return None
    _, _, vh = torch.linalg.svd(v, full_matrices=False)
    b = vh[-1]
    # The null-space vector carries an arbitrary sign; B = K^-T K^-1 has B33 > 0 for a real camera.
    if b[5] < 0:
        b = -b
    b11, b12, b22, b13, b23, b33 = b.unbind()
    # Closed-form intrinsic recovery (zero skew), Zhang (2000) appendix B.
    cy = (b12 * b13 - b11 * b23) / (b11 * b22 - b12 * b12)
    lam = b33 - (b13 * b13 + cy * (b12 * b13 - b11 * b23)) / b11
    if not bool(torch.isfinite(lam)) or lam <= 0:
        return None
    fx = torch.sqrt(lam / b11)
    fy = torch.sqrt(lam * b11 / (b11 * b22 - b12 * b12))
    cx = -b13 / b11
    intrinsics = torch.stack([fx, fy, cx, cy])
    if not bool(torch.isfinite(intrinsics).all()) or bool((intrinsics[:2] <= 0).any()):
        return None
    return intrinsics


def _pose_from_homography(
    h: torch.Tensor, intrinsic_params: torch.Tensor, world_points: Optional[torch.Tensor] = None
) -> Optional[torch.Tensor]:
    """Decompose a board-to-image homography into the view's world-to-camera Rodrigues pose.

    The homography is defined up to sign, so the candidate with the board points in front of the camera
    (positive camera-frame depth, the cheirality constraint) is selected.
    """
    fx, fy, cx, cy = intrinsic_params.unbind()
    k_inv = torch.tensor(
        [[1.0 / fx, 0.0, -cx / fx], [0.0, 1.0 / fy, -cy / fy], [0.0, 0.0, 1.0]],
        dtype=intrinsic_params.dtype,
        device=intrinsic_params.device,
    )
    a = k_inv @ h  # columns are scaled rotation columns and translation
    scale = 1.0 / torch.linalg.vector_norm(a[:, 0]).clamp_min(torch.finfo(a.dtype).eps)
    r1 = a[:, 0] * scale
    r2 = a[:, 1] * scale
    t = a[:, 2] * scale
    r3 = torch.linalg.cross(r1, r2)
    rotation = torch.stack([r1, r2, r3], dim=-1)
    # Project onto SO(3) with a polar factor to clean up the noise of the homography estimate.
    u, _, vh = torch.linalg.svd(rotation)
    rotation = u @ vh
    if bool(torch.det(rotation) < 0):
        u = u.clone()
        u[:, -1] = -u[:, -1]
        rotation = u @ vh
    # Resolve the homography sign: board points must lie in front of the camera (z > 0).
    if world_points is not None:
        depths = (rotation @ world_points.transpose(-1, -2) + t.unsqueeze(-1))[2]
        if bool(torch.median(depths) < 0):
            rotation = -rotation
            t = -t
    elif t[2] < 0:
        rotation = -rotation
        t = -t
    axis_angle = rotation_matrix_to_axis_angle(rotation)
    if not bool(torch.isfinite(axis_angle).all() and torch.isfinite(t).all()):
        return None
    return torch.cat([axis_angle, t])


def zhang_reestimate(
    world_points: Sequence[torch.Tensor],
    image_points: Sequence[torch.Tensor],
    inlier_masks: Optional[Sequence[torch.Tensor]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    r"""Re-estimate intrinsics and per-view poses from (ideally undistorted) planar pixels via Zhang homographies.

    Args:
        world_points: Per-view planar world points, each ``(N_m, 3)``.
        image_points: Per-view (undistorted) pixels, each ``(N_m, 2)``.
        inlier_masks: Optional per-view bool masks; only masked points enter the homographies.

    Returns:
        Tuple ``(intrinsic_params, pose_params)``; either is None when the homographies or the closed-form
        extraction are degenerate.
    """
    homographies = []
    for k, (wpts, ipts) in enumerate(zip(world_points, image_points)):
        if inlier_masks is not None and inlier_masks[k] is not None:
            mask = inlier_masks[k]
            if int(mask.sum()) < MIN_POINTS_PER_VIEW:
                return None, None
            wpts, ipts = wpts[mask], ipts[mask]
        board_xy = _planar_xy(wpts)
        h = _homography_view(board_xy, ipts, None)
        if not bool(torch.isfinite(h).all()):
            return None, None
        homographies.append(h)
    intrinsic_params = _intrinsics_from_homographies(torch.stack(homographies, dim=0))
    if intrinsic_params is None:
        return None, None
    poses = []
    for k, h in enumerate(homographies):
        wpts = world_points[k] if inlier_masks is None or inlier_masks[k] is None else world_points[k][inlier_masks[k]]
        pose = _pose_from_homography(h, intrinsic_params, wpts)
        if pose is None:
            return intrinsic_params, None
        poses.append(pose)
    return intrinsic_params, torch.stack(poses, dim=0)


def _nominal_intrinsics(
    image_points: Sequence[torch.Tensor],
    image_size: Optional[Tuple[int, int]],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Rough pinhole guess: centered principal point and a focal length of the largest image extent."""
    if image_size is not None:
        height, width = image_size
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        focal = float(max(width, height))
    else:
        all_points = torch.cat(image_points, dim=0)
        mins = all_points.amin(dim=0)
        maxs = all_points.amax(dim=0)
        center = (mins + maxs) / 2.0
        cx, cy = center[0].item(), center[1].item()
        focal = float((maxs - mins).max().clamp_min(1.0))
    return torch.tensor([focal, focal, cx, cy], dtype=dtype, device=device)


def _pnp_poses(
    world_points: Sequence[torch.Tensor],
    image_points: Sequence[torch.Tensor],
    intrinsic_params: torch.Tensor,
    view_valid: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], List[Optional[str]]]:
    """Per-view world-to-camera poses through the DLT PnP on valid views."""
    k = torch.zeros((3, 3), dtype=intrinsic_params.dtype, device=intrinsic_params.device)
    fx, fy, cx, cy = intrinsic_params.unbind()
    k[0, 0], k[1, 1], k[0, 2], k[1, 2], k[2, 2] = fx, fy, cx, cy, 1.0
    poses: List[Optional[torch.Tensor]] = [None] * len(world_points)
    reasons: List[Optional[str]] = [None] * len(world_points)
    for i in range(len(world_points)):
        if not bool(view_valid[i]):
            continue
        try:
            rt = solve_pnp_dlt(world_points[i][None], image_points[i][None], k[None])[0]
        except (AssertionError, RuntimeError) as exc:
            reasons[i] = f"degenerate_pnp: {type(exc).__name__}"
            view_valid[i] = False
            continue
        axis_angle = rotation_matrix_to_axis_angle(rt[:, :3])
        if not bool(torch.isfinite(axis_angle).all() and torch.isfinite(rt[:, 3]).all()):
            reasons[i] = "degenerate_pnp: non-finite pose"
            view_valid[i] = False
            continue
        poses[i] = torch.cat([axis_angle, rt[:, 3]])
    if not bool(view_valid.any()):
        return None, reasons
    pose_params = torch.stack([p for p in poses if p is not None], dim=0)
    return pose_params, reasons


def _validate_views(
    world_points: Sequence[torch.Tensor],
    image_points: Sequence[torch.Tensor],
    weights: Optional[Sequence[Optional[torch.Tensor]]],
    image_size: Optional[Tuple[int, int]],
) -> Tuple[torch.Tensor, List[Optional[str]]]:
    """Per-view input and coverage checks, returning the validity mask and per-view reason strings."""
    num_views = len(world_points)
    device = image_points[0].device
    view_valid = torch.ones(num_views, dtype=torch.bool, device=device)
    view_reasons: List[Optional[str]] = [None] * num_views
    for i in range(num_views):
        n = image_points[i].shape[0]
        if n < MIN_POINTS_PER_VIEW:
            view_valid[i] = False
            view_reasons[i] = f"fewer_than_four_points: view {i} has {n} points"
            continue
        if not bool(torch.isfinite(world_points[i]).all() and torch.isfinite(image_points[i]).all()):
            view_valid[i] = False
            view_reasons[i] = f"non_finite: view {i} contains non-finite coordinates"
            continue
        if weights is not None and weights[i] is not None and not bool(weights[i].any()):
            view_valid[i] = False
            view_reasons[i] = f"all_weights_zero: view {i}"
            continue
        if not _point_spread_ok(image_points[i], image_size):
            view_valid[i] = False
            view_reasons[i] = f"degenerate_coverage: view {i} points do not span the image"
    return view_valid, view_reasons


def _planar_initial_estimate(
    world_points: Sequence[torch.Tensor],
    image_points: Sequence[torch.Tensor],
    weights: Optional[Sequence[Optional[torch.Tensor]]],
    image_size: Optional[Tuple[int, int]],
    view_valid: torch.Tensor,
    view_reasons: List[Optional[str]],
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[Optional[torch.Tensor]], bool, Optional[str]]:
    """Robust homography-based Zhang initialization for a planar board."""
    num_views = len(world_points)
    homographies: List[Optional[torch.Tensor]] = [None] * num_views
    inlier_masks: List[Optional[torch.Tensor]] = [None] * num_views
    valid_indices = [i for i in range(num_views) if bool(view_valid[i])]
    for i in valid_indices:
        board_xy = _planar_xy(world_points[i])
        # RANSAC keeps gross mismatches out of both the homography and the downstream fit; a fixed seed keeps
        # the calibration deterministic. The threshold is in pixels.
        h, mask = _robust_homography_view(board_xy, image_points[i], 3.0)
        inlier_fraction = 0.0 if mask is None else float(mask.float().mean())
        too_few_inliers = int(mask.sum() if mask is not None else 0) < MIN_POINTS_PER_VIEW
        if h is None or mask is None or too_few_inliers or inlier_fraction < _MIN_INLIER_FRACTION:
            view_valid[i] = False
            view_reasons[i] = (
                f"degenerate_homography: view {i} robust homography supported by "
                f"{inlier_fraction:.2f} of the points, below {_MIN_INLIER_FRACTION:.2f}"
            )
            continue
        if weights is not None and weights[i] is not None:
            mask = mask & (weights[i] > 0)
        homographies[i] = h
        inlier_masks[i] = mask
    valid_indices = [i for i in range(num_views) if bool(view_valid[i])]
    if len(valid_indices) < 2:
        return None, None, inlier_masks, True, "fewer than two views support a planar homography"
    intrinsic_params = _intrinsics_from_homographies(
        torch.stack([homographies[i] for i in valid_indices], dim=0)
    )
    if intrinsic_params is None:
        # Fall back to a nominal intrinsic guess and PnP poses rather than failing on the closed form.
        intrinsic_params = _nominal_intrinsics(
            [image_points[i] for i in valid_indices], image_size, dtype, device
        )
        pose_params, pnp_reasons = _pnp_poses(world_points, image_points, intrinsic_params, view_valid)
        for i, reason in enumerate(pnp_reasons):
            if reason is not None:
                view_reasons[i] = reason
        return intrinsic_params, pose_params, inlier_masks, True, None
    poses: List[Optional[torch.Tensor]] = [None] * num_views
    for i in valid_indices:
        pose = _pose_from_homography(homographies[i], intrinsic_params, world_points[i])
        if pose is None:
            view_valid[i] = False
            view_reasons[i] = f"degenerate_homography: view {i} pose decomposition failed"
            continue
        poses[i] = pose
    valid_indices = [i for i in range(num_views) if bool(view_valid[i])]
    if len(valid_indices) < 2:
        return None, None, inlier_masks, True, "fewer than two views yielded a valid pose"
    pose_params = torch.stack([poses[i] for i in valid_indices], dim=0)
    return intrinsic_params, pose_params, inlier_masks, True, None


def _filter_stationary_views(
    pose_params: torch.Tensor,
    valid_indices: Sequence[int],
    view_valid: torch.Tensor,
    view_reasons: List[Optional[str]],
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, List[int], Optional[str]]:
    """Drop redundant views whose camera barely moved relative to the batch median."""
    translations = pose_params[:, 3:6]
    median_t = translations.median(dim=0).values
    distances = torch.linalg.vector_norm(translations - median_t, dim=-1)
    scale = torch.linalg.vector_norm(translations, dim=-1).median().clamp_min(torch.finfo(dtype).eps)
    stationary = distances < MIN_BASELINE_RATIO * scale
    if not bool(stationary.sum() > 1):
        return pose_params, list(valid_indices), None
    kept_rows: List[int] = []
    median_kept = False
    for local, i in enumerate(valid_indices):
        if bool(stationary[local]):
            if not median_kept:
                median_kept = True  # the first stationary view anchors the cluster and stays
                kept_rows.append(local)
            else:
                view_valid[i] = False
                view_reasons[i] = f"small_baseline: view {i} camera barely moved relative to the other views"
        else:
            kept_rows.append(local)
    surviving = [i for i in valid_indices if bool(view_valid[i])]
    if len(surviving) < 2:
        return pose_params, list(surviving), "fewer than two views remain after the baseline check"
    return pose_params[kept_rows], surviving, None


def initial_estimate(
    world_points: Sequence[torch.Tensor],
    image_points: Sequence[torch.Tensor],
    weights: Optional[Sequence[Optional[torch.Tensor]]],
    image_size: Optional[Tuple[int, int]],
) -> InitialEstimate:
    r"""Build the initial camera parameters, per-view poses and per-view validity flags.

    Args:
        world_points: Per-view world points, each ``(N_m, 3)``.
        image_points: Per-view measured pixels, each ``(N_m, 2)``.
        weights: Per-view optional non-negative weights, each ``(N_m,)``.
        image_size: Optional ``(height, width)`` for the frame-coverage check.

    Returns:
        :class:`InitialEstimate`; its ``fatal_reason`` is set when no view can start a fit.
    """
    num_views = len(world_points)
    dtype = _at_least_float32(image_points[0]).dtype
    device = image_points[0].device
    view_valid, view_reasons = _validate_views(world_points, image_points, weights, image_size)

    if not bool(view_valid.any()):
        return InitialEstimate(None, None, view_valid, view_reasons, False, "no usable view passed the input checks")

    valid_indices = [i for i in range(num_views) if bool(view_valid[i])]
    is_planar = _world_planarity([world_points[i] for i in valid_indices])

    inlier_masks: List[Optional[torch.Tensor]]
    if is_planar:
        intrinsic_params, pose_params, inlier_masks, is_planar, fatal = _planar_initial_estimate(
            world_points, image_points, weights, image_size, view_valid, view_reasons, dtype, device
        )
        if fatal is not None:
            return InitialEstimate(None, None, view_valid, view_reasons, True, fatal, None)
    else:
        intrinsic_params = _nominal_intrinsics(
            [image_points[i] for i in valid_indices], image_size, dtype, device
        )
        pose_params, pnp_reasons = _pnp_poses(world_points, image_points, intrinsic_params, view_valid)
        for i, reason in enumerate(pnp_reasons):
            if reason is not None:
                view_reasons[i] = reason
        # Non-planar initialization has no per-view robust mask; every finite, weighted point starts as an inlier.
        inlier_masks = [
            None if not bool(view_valid[i])
            else (torch.ones(image_points[i].shape[0], dtype=torch.bool, device=device)
                  if weights is None or weights[i] is None
                  else weights[i] > 0)
            for i in range(num_views)
        ]
        if pose_params is None:
            return InitialEstimate(None, None, view_valid, view_reasons, False, "no view yielded a valid PnP pose")

    if pose_params is None:
        return InitialEstimate(
            None, None, view_valid, view_reasons, is_planar, "no view yielded an initial pose"
        )

    valid_indices = [i for i in range(num_views) if bool(view_valid[i])]
    pose_params, valid_indices, fatal = _filter_stationary_views(
        pose_params, valid_indices, view_valid, view_reasons, dtype
    )
    if fatal is not None:
        return InitialEstimate(None, None, view_valid, view_reasons, is_planar, fatal, None)

    # Align the masks to the same compressed valid-view order as pose_params.
    aligned_masks = [inlier_masks[i] for i in valid_indices]
    return InitialEstimate(
        intrinsic_params, pose_params, view_valid, view_reasons, is_planar, None, aligned_masks
    )
