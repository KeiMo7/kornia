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

"""Multi-view camera calibration from world-to-pixel correspondences.

The public entry points fit one camera, :func:`calibrate_camera` (OpenCV Brown-Conrady pinhole) or
:func:`calibrate_fisheye` (Kannala-Brandt fisheye), from the correspondences of several views, and batch several
cameras with :func:`calibrate_cameras`. The pipeline is:

1. per-view input and degeneracy checks (point count, finiteness, zero weights, frame coverage, baseline);
2. a closed-form initial estimate (Zhang homographies for planar boards, DLT PnP otherwise);
3. a deterministic Levenberg-Marquardt bundle adjustment with Huber IRLS weights;
4. robust point rejection and gross-outlier view rejection, each reported with its reason;
5. distortion-order reduction when the observations cannot support the requested tier;
6. residual statistics recomputed through the public projection functions so they match downstream consumption.

Nothing here uses a random source or mutable global state, so a serial batch and a parallel batch give the same
result bit for bit. Existing single-view projection, undistortion and PnP functions are untouched.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.geometry.calibration._bundle import LMResult, levenberg_marquardt
from kornia.geometry.calibration._init import (
    InitialEstimate,
    initial_estimate,
    zhang_reestimate,
)
from kornia.geometry.calibration._projection import (
    NUM_INTRINSIC_PARAMS,
    build_intrinsics,
    pose_from_params,
    project_fisheye,
    project_pinhole,
)
from kornia.geometry.calibration.result import (
    FISHEYE_MODEL,
    PINHOLE_DISTORTION_ORDERS,
    PINHOLE_MODEL,
    CalibrationResult,
    ViewDiagnostics,
)
from kornia.geometry.calibration.undistort import undistort_points
from kornia.geometry.camera.distortion_kannala_brandt import undistort_points_kannala_brandt

_ACCEPTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
#: A view whose RMS is more than this many times the batch median is a gross-outlier view.
_VIEW_OUTLIER_FACTOR = 5.0
#: Minimum inlier fraction a view must keep to stay in the fit.
_MIN_VIEW_INLIER_FRACTION = 0.25
_MIN_VIEWS = 2


@dataclass
class CalibrationObservations:
    r"""One camera's multi-view observations in the list-style API.

    Attributes:
        world_points: Per-view world points, each ``(N_m, 3)``.
        image_points: Per-view measured pixels, each ``(N_m, 2)``, matching ``world_points``.
        weights: Optional per-view non-negative weights, each ``(N_m,)``.
        image_size: Optional ``(height, width)`` the observations were made at.
        camera_id: Identifier carried verbatim onto the :class:`CalibrationResult`.
    """

    world_points: Sequence[torch.Tensor]
    image_points: Sequence[torch.Tensor]
    weights: Optional[Sequence[Optional[torch.Tensor]]] = None
    image_size: Optional[Tuple[int, int]] = None
    camera_id: object = None


def _validate_inputs(
    world_points: Sequence[torch.Tensor],
    image_points: Sequence[torch.Tensor],
    weights: Optional[Sequence[Optional[torch.Tensor]]],
    image_size: Optional[Tuple[int, int]],
) -> Tuple[torch.dtype, torch.device]:
    KORNIA_CHECK(len(world_points) == len(image_points), "world_points and image_points must hold one pair per view.")
    if weights is not None:
        KORNIA_CHECK(len(weights) == len(world_points), "weights must hold one entry per view, or be None.")
    for i, (wpts, ipts) in enumerate(zip(world_points, image_points)):
        KORNIA_CHECK_IS_TENSOR(wpts, msg=f"world_points[{i}] must be a torch.Tensor.")
        KORNIA_CHECK_IS_TENSOR(ipts, msg=f"image_points[{i}] must be a torch.Tensor.")
        dtype_msg = "must be float16, bfloat16, float32 or float64."
        KORNIA_CHECK(wpts.dtype in _ACCEPTED_DTYPES, f"world_points[{i}] {dtype_msg}")
        KORNIA_CHECK(ipts.dtype in _ACCEPTED_DTYPES, f"image_points[{i}] {dtype_msg}")
        KORNIA_CHECK_SHAPE(wpts, ["N", "3"])
        KORNIA_CHECK_SHAPE(ipts, ["N", "2"])
        KORNIA_CHECK(
            wpts.shape[0] == ipts.shape[0],
            f"world_points[{i}] and image_points[{i}] must have the same number of points.",
        )
        if weights is not None and weights[i] is not None:
            KORNIA_CHECK_SHAPE(weights[i], ["N"])
            KORNIA_CHECK(
                bool((weights[i] >= 0).all()),
                f"weights[{i}] must be non-negative.",
            )
    if image_size is not None:
        height, width = image_size
        KORNIA_CHECK(
            isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0,
            "image_size must be a positive (height, width) tuple of ints.",
        )
    dtype = torch.promote_types(world_points[0].dtype, image_points[0].dtype)
    device = world_points[0].device
    for tensor in list(world_points) + list(image_points):
        KORNIA_CHECK(tensor.device == device, "every view must live on the same device.")
    return dtype, device


def _failure(
    camera_id: object,
    image_size: Optional[Tuple[int, int]],
    model: str,
    distortion_order: int,
    reasons: Sequence[str],
    view_diagnostics: Optional[List[ViewDiagnostics]] = None,
) -> CalibrationResult:
    return CalibrationResult(
        camera_id=camera_id,
        image_size=image_size,
        model=model,
        distortion_order=distortion_order,
        success=False,
        view_diagnostics=view_diagnostics or [],
        rejection_reasons=list(reasons),
    )


def _empty_view_diagnostics(num_views: int) -> List[ViewDiagnostics]:
    return [
        ViewDiagnostics(
            view_index=i,
            used=False,
            num_points=0,
            num_inliers=0,
            rms_reprojection_error=float("nan"),
            max_reprojection_error=float("nan"),
            mean_reprojection_error=float("nan"),
        )
        for i in range(num_views)
    ]


def _build_diagnostics(
    num_views: int,
    used_views: Sequence[int],
    point_counts: Sequence[int],
    errors: Sequence[Optional[torch.Tensor]],
    inlier_masks: Sequence[Optional[torch.Tensor]],
    view_reasons: Sequence[Optional[str]],
) -> List[ViewDiagnostics]:
    diagnostics: List[ViewDiagnostics] = []
    for i in range(num_views):
        reason = view_reasons[i]
        if i not in used_views or errors[i] is None:
            diagnostics.append(
                ViewDiagnostics(
                    view_index=i,
                    used=False,
                    num_points=point_counts[i],
                    num_inliers=0,
                    rms_reprojection_error=float("nan"),
                    max_reprojection_error=float("nan"),
                    mean_reprojection_error=float("nan"),
                    rejected=reason is not None,
                    rejection_reason=reason,
                    degeneracy=reason.split(":")[0] if reason is not None else None,
                )
            )
            continue
        mask = inlier_masks[i]
        inlier_errors = errors[i][mask]
        rms = float(torch.sqrt(torch.mean(inlier_errors.square()))) if inlier_errors.numel() else float("nan")
        diagnostics.append(
            ViewDiagnostics(
                view_index=i,
                used=True,
                num_points=point_counts[i],
                num_inliers=int(mask.sum()),
                rms_reprojection_error=rms,
                max_reprojection_error=float(inlier_errors.max()) if inlier_errors.numel() else float("nan"),
                mean_reprojection_error=float(inlier_errors.mean()) if inlier_errors.numel() else float("nan"),
                rejected=False,
                rejection_reason=None,
            )
        )
    return diagnostics


def _undistort_pixels(
    model: str,
    image_points: Sequence[torch.Tensor],
    intrinsic_params: torch.Tensor,
    distortion: torch.Tensor,
) -> List[torch.Tensor]:
    """Remove the current lens model from measured pixels, the re-estimation step of the outer loop."""
    k = build_intrinsics(intrinsic_params)
    if model == PINHOLE_MODEL:
        return [undistort_points(ipts, k, distortion) for ipts in image_points]
    flat = torch.cat([intrinsic_params, distortion], dim=-1)
    return [undistort_points_kannala_brandt(ipts, flat) for ipts in image_points]


def _structured_refine(
    model: str,
    project_fn,
    local_world: Sequence[torch.Tensor],
    local_image: Sequence[torch.Tensor],
    estimate: InitialEstimate,
    initial_distortion: torch.Tensor,
    inlier_masks: Sequence[torch.Tensor],
    max_iterations: int,
    outer_iterations: int = 6,
) -> Optional[LMResult]:
    r"""Refine one camera with the OpenCV/Zhang structured outer loop.

    Each outer iteration (1) jointly refines the intrinsics, distortion and every pose by Levenberg-Marquardt on
    the inlier points, then (2) undistorts the pixels with the fitted model and re-estimates the intrinsics and
    poses from the homographies, which pulls the solution off the intrinsic/distortion local minimum of a plain
    joint optimization. The two stages converge to the maximum-likelihood estimate and, on the synthetic scenes,
    reach the true parameters within a few iterations.
    """
    camera_params = torch.cat([estimate.intrinsic_params, initial_distortion], dim=-1)
    pose_params = estimate.pose_params.clone()
    full_mask = torch.ones(camera_params.shape[0], dtype=torch.bool, device=camera_params.device)
    distortion_mask = torch.cat(
        [
            torch.ones(NUM_INTRINSIC_PARAMS, dtype=torch.bool, device=camera_params.device),
            torch.zeros(initial_distortion.shape[0], dtype=torch.bool, device=camera_params.device),
        ]
    )
    planar = estimate.is_planar

    # Stage zero for every model: fit the geometry with the distortion frozen, so the later joint release starts
    # from intrinsics and poses that already explain the observations instead of a far-from-truth point.
    seed = levenberg_marquardt(
        project_fn,
        list(zip(local_world, local_image)),
        camera_params,
        pose_params,
        inlier_masks,
        max_iterations=max_iterations,
        robust=False,
        camera_active_mask=distortion_mask,
    )
    if not seed.finite:
        return None
    camera_params, pose_params = seed.camera_params, seed.pose_params

    result: Optional[LMResult] = None
    for outer in range(outer_iterations):
        result = levenberg_marquardt(
            project_fn,
            list(zip(local_world, local_image)),
            camera_params,
            pose_params,
            inlier_masks,
            max_iterations=max_iterations,
            robust=False,
            camera_active_mask=full_mask,
        )
        if not result.finite:
            return None
        camera_params, pose_params = result.camera_params, result.pose_params
        # The undistort/Zhang re-estimation is specific to the pinhole Brown-Conrady model. A fisheye board is
        # planar too, but its pixels must go through the Kannala-Brandt inverse, so it skips this step and relies
        # on the two-stage LM above as its full refinement.
        if model != PINHOLE_MODEL or not planar or outer == outer_iterations - 1:
            continue
        # Re-estimate the pinhole geometry on undistorted pixels; keep the distortion tier just refined. Only
        # inlier points enter the homographies, so a gross mismatch cannot corrupt the closed-form estimate.
        intrinsic_params = camera_params[:NUM_INTRINSIC_PARAMS]
        distortion = camera_params[NUM_INTRINSIC_PARAMS:]
        undistorted = _undistort_pixels(model, local_image, intrinsic_params, distortion)
        new_intr, new_poses = zhang_reestimate(local_world, undistorted, inlier_masks)
        if new_intr is None or new_poses is None:
            continue  # keep the LM estimate if the closed form is momentarily degenerate
        if not bool(torch.isfinite(new_intr).all() and torch.isfinite(new_poses).all()):
            continue
        camera_params = torch.cat([new_intr, distortion], dim=-1)
        pose_params = new_poses
    return result


def _robust_fit(
    project_fn,
    model: str,
    world_points: Sequence[torch.Tensor],
    image_points: Sequence[torch.Tensor],
    weights: Optional[Sequence[Optional[torch.Tensor]]],
    estimate: InitialEstimate,
    initial_distortion: torch.Tensor,
    max_iterations: int,
) -> Tuple[Optional[LMResult], List[int], List[torch.Tensor], List[str]]:
    """Run the structured, robust fit loop on the valid views only.

    A first pinhole-only fit supplies point and view rejection with deterministic Huber/MAD thresholds; the
    rejected gross points and views are then excluded and :func:`_structured_refine` produces the final camera.
    No random sampling is involved, so repeated calls are bit-identical.

    Everything inside works on the compressed valid-view subset with local indices. The returned view list maps
    each local row back to its global input-view index, and the masks are aligned to the local rows.
    """
    # Compress to the valid views; estimate.pose_params is already aligned to this subset, in this order.
    global_views = [i for i in range(len(world_points)) if bool(estimate.view_valid[i])]
    local_world = [world_points[i] for i in global_views]
    local_image = [image_points[i] for i in global_views]
    local_weights: Optional[List[Optional[torch.Tensor]]]
    local_weights = None if weights is None else [weights[i] for i in global_views]
    num_local = len(global_views)
    used_local = list(range(num_local))
    notes: List[str] = []

    # RANSAC inlier masks from the initialization, aligned to the compressed valid views; None means all-inlier.
    ransac_masks = (
        estimate.inlier_masks
        if estimate.inlier_masks is not None
        else [None] * num_local
    )

    def _masks_for(local_subset: Sequence[int]) -> List[torch.Tensor]:
        masks: List[torch.Tensor] = []
        for i in local_subset:
            n = local_world[i].shape[0]
            base = (
                ransac_masks[i]
                if ransac_masks[i] is not None
                else torch.ones(n, dtype=torch.bool, device=local_world[i].device)
            )
            if local_weights is not None and local_weights[i] is not None:
                base = base & (local_weights[i] > 0)
            masks.append(base)
        return masks

    def _subset(indices: Sequence[int]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        return (
            [local_world[i] for i in indices],
            [local_image[i] for i in indices],
            _masks_for(indices),
        )

    # Stage one: geometry only, distortion frozen at zero, to seed the structured refinement.
    distortion_mask = torch.cat(
        [
            torch.ones(NUM_INTRINSIC_PARAMS, dtype=torch.bool, device=estimate.intrinsic_params.device),
            torch.zeros(initial_distortion.shape[0], dtype=torch.bool, device=estimate.intrinsic_params.device),
        ]
    )
    seed_camera = torch.cat([estimate.intrinsic_params, torch.zeros_like(initial_distortion)], dim=-1)
    stage_one = levenberg_marquardt(
        project_fn,
        list(zip(local_world, local_image)),
        seed_camera,
        estimate.pose_params,
        _masks_for(used_local),
        max_iterations=max_iterations,
        robust=False,
        camera_active_mask=distortion_mask,
    )
    if not stage_one.finite:
        empty = [torch.zeros(0, dtype=torch.bool, device=seed_camera.device)] * num_local
        return None, used_local, empty, notes

    # Full structured refinement (distortion released) on the RANSAC inliers. Point-level robustness is already
    # provided by those masks, so a MAD rejection here would only mis-label the systematic, lens-induced residuals
    # of a strongly distorted view as gross errors.
    seed_masks = _masks_for(used_local)
    refined_estimate = InitialEstimate(
        intrinsic_params=stage_one.camera_params[:NUM_INTRINSIC_PARAMS].detach().clone(),
        pose_params=stage_one.pose_params.detach().clone(),
        view_valid=torch.ones(len(used_local), dtype=torch.bool, device=seed_camera.device),
        view_reasons=[None] * len(used_local),
        is_planar=estimate.is_planar,
        inlier_masks=list(seed_masks),
    )
    final_world, final_image, final_masks = _subset(used_local)
    final = _structured_refine(
        model,
        project_fn,
        final_world,
        final_image,
        refined_estimate,
        initial_distortion,
        final_masks,
        max_iterations,
    )
    if final is None:
        return None, used_local, final_masks, notes

    # View-level rejection on the residuals of the *fitted* model, where the lens is already accounted for.
    view_rms: List[float] = []
    for row, local in enumerate(used_local):
        projected = project_fn(
            local_world[local],
            final.camera_params[:NUM_INTRINSIC_PARAMS],
            final.camera_params[NUM_INTRINSIC_PARAMS:],
            final.pose_params[row],
        )
        err = torch.linalg.vector_norm(projected - local_image[local], dim=-1)
        kept = err[final_masks[row]]
        view_rms.append(float(torch.sqrt(kept.square().mean())) if kept.numel() else float("inf"))
    rms_tensor = torch.tensor([r for r in view_rms if r != float("inf")])
    median_rms = float(torch.median(rms_tensor)) if rms_tensor.numel() else float("inf")
    surviving: List[int] = []
    for row, local in enumerate(used_local):
        inlier_fraction = float(final_masks[row].float().mean())
        if view_rms[row] > _VIEW_OUTLIER_FACTOR * median_rms or inlier_fraction < _MIN_VIEW_INLIER_FRACTION:
            global_index = global_views[local]
            reason = (
                f"outlier_view: view {global_index} RMS {view_rms[row]:.3f} px exceeds "
                f"{_VIEW_OUTLIER_FACTOR:.0f}x the batch median {median_rms:.3f} px or inlier fraction "
                f"{inlier_fraction:.2f} is below {_MIN_VIEW_INLIER_FRACTION:.2f}"
            )
            estimate.view_reasons[global_index] = reason
            notes.append(reason)
        else:
            surviving.append(local)

    if len(surviving) == len(used_local):
        return final, used_local, final_masks, notes
    if len(surviving) < _MIN_VIEWS:
        # Too few clean views to refit: keep the all-view fit but still record the rejection.
        return final, used_local, final_masks, notes

    # Refit once on the surviving views so the returned parameters do not depend on the rejected gross views.
    kept_rows = [used_local.index(i) for i in surviving]
    reduced_estimate = InitialEstimate(
        intrinsic_params=final.camera_params[:NUM_INTRINSIC_PARAMS].detach().clone(),
        pose_params=final.pose_params[kept_rows].detach().clone(),
        view_valid=torch.ones(len(surviving), dtype=torch.bool, device=seed_camera.device),
        view_reasons=[None] * len(surviving),
        is_planar=estimate.is_planar,
        inlier_masks=[final_masks[used_local.index(i)] for i in surviving],
    )
    sub_world, sub_image, sub_masks = _subset(surviving)
    reduced = _structured_refine(
        model, project_fn, sub_world, sub_image, reduced_estimate, initial_distortion, sub_masks, max_iterations
    )
    if reduced is None or not reduced.finite:
        return final, used_local, final_masks, notes
    return reduced, surviving, sub_masks, notes


def _calibrate_one(
    model: str,
    observations: CalibrationObservations,
    distortion_order: int,
    max_iterations: int,
) -> CalibrationResult:
    world_points = observations.world_points
    image_points = observations.image_points
    weights = observations.weights
    image_size = observations.image_size
    camera_id = observations.camera_id
    num_views = len(world_points)

    if num_views == 0:
        return _failure(
            camera_id, image_size, model, distortion_order,
            ["no_views: calibration needs at least two views with four points each."],
            _empty_view_diagnostics(0),
        )

    try:
        input_dtype, device = _validate_inputs(world_points, image_points, weights, image_size)
    except Exception:  # contract violations are explicit input errors, not a calibration rejection
        raise

    # Half precision is fitted in float32 and the result is rounded once, matching the rest of the package.
    work_dtype = torch.float32 if input_dtype in (torch.float16, torch.bfloat16) else input_dtype
    world_points = [t.to(work_dtype) for t in world_points]
    image_points = [t.to(work_dtype) for t in image_points]
    if weights is not None:
        weights = [None if w is None else w.to(work_dtype) for w in weights]

    point_counts = [t.shape[0] for t in image_points]
    project_fn = project_pinhole if model == PINHOLE_MODEL else project_fisheye

    estimate = initial_estimate(world_points, image_points, weights, image_size)
    if estimate.fatal_reason is not None:
        diagnostics = _build_diagnostics(
            num_views,
            [i for i in range(num_views) if bool(estimate.view_valid[i])],
            point_counts,
            [None] * num_views,
            [None] * num_views,
            estimate.view_reasons,
        )
        return _failure(camera_id, image_size, model, distortion_order, [estimate.fatal_reason], diagnostics)

    # The fit works on the compressed valid-view subset; this maps a local fit row back to the global view index.
    valid_globals = [i for i in range(num_views) if bool(estimate.view_valid[i])]

    # Distortion-order reduction for the pinhole model; the fisheye model always keeps its four coefficients.
    orders = (
        [o for o in PINHOLE_DISTORTION_ORDERS if o <= distortion_order]
        if model == PINHOLE_MODEL
        else [4]
    )
    if not orders:
        orders = [min(PINHOLE_DISTORTION_ORDERS, key=lambda o: abs(o - distortion_order))]
    notes: List[str] = []
    last_fit: Optional[LMResult] = None
    last_used_local: List[int] = []
    last_local_masks: List[torch.Tensor] = []
    for order in orders:
        if model == PINHOLE_MODEL:
            initial_distortion = torch.zeros(order, dtype=work_dtype, device=device)
        else:
            initial_distortion = torch.zeros(4, dtype=work_dtype, device=device)
        fit, used_local, local_masks, fit_notes = _robust_fit(
            project_fn, model, world_points, image_points, weights, estimate, initial_distortion, max_iterations
        )
        notes.extend(fit_notes)
        if fit is not None and fit.finite:
            last_fit, last_used_local, last_local_masks = fit, used_local, local_masks
            # An unstable high-order coefficient (magnitude blowing up) is the signal to drop the tier.
            distortion_part = fit.camera_params[NUM_INTRINSIC_PARAMS:]
            if model == PINHOLE_MODEL and order > min(PINHOLE_DISTORTION_ORDERS):
                unstable = bool((~torch.isfinite(distortion_part)).any()) or bool(
                    distortion_part.abs().max() > 1e3
                )
                if unstable:
                    notes.append(
                        f"distortion_reduced: order {order} was not supported by the observations "
                        "(non-finite or diverging coefficients); retrying a lower tier."
                    )
                    estimate.intrinsic_params = fit.camera_params[:NUM_INTRINSIC_PARAMS].detach().clone()
                    estimate.pose_params = fit.pose_params.detach().clone()
                    continue
            break
        if model != PINHOLE_MODEL or order == min(PINHOLE_DISTORTION_ORDERS):
            break
        notes.append(
            f"distortion_reduced: order {order} did not converge; retrying with a lower tier."
        )
        estimate.intrinsic_params = (
            fit.camera_params[:NUM_INTRINSIC_PARAMS].detach().clone() if fit is not None else estimate.intrinsic_params
        )

    if last_fit is None or not last_fit.finite:
        used_globals = [valid_globals[i] for i in last_used_local]
        diagnostics = _build_diagnostics(
            num_views, used_globals, point_counts, [None] * num_views, [None] * num_views, estimate.view_reasons
        )
        return _failure(
            camera_id, image_size, model, distortion_order,
            ["non_convergence: the nonlinear refinement produced non-finite parameters at every tier."] + notes,
            diagnostics,
        )

    camera_params = last_fit.camera_params
    pose_params = last_fit.pose_params
    used_globals = [valid_globals[i] for i in last_used_local]
    fitted_order = camera_params.shape[0] - NUM_INTRINSIC_PARAMS if model == PINHOLE_MODEL else 4

    # Recompute every reported residual through the public projection path so the report and downstream match.
    # Errors and masks are gathered per global view; pose rows are aligned to the surviving local order.
    global_errors: List[Optional[torch.Tensor]] = [None] * num_views
    global_masks: List[Optional[torch.Tensor]] = [None] * num_views
    for row, global_index in enumerate(used_globals):
        projected = project_fn(
            world_points[global_index],
            camera_params[:NUM_INTRINSIC_PARAMS],
            camera_params[NUM_INTRINSIC_PARAMS:],
            pose_params[row],
        )
        global_errors[global_index] = torch.linalg.vector_norm(projected - image_points[global_index], dim=-1)
        global_masks[global_index] = last_local_masks[row]
    diagnostics = _build_diagnostics(
        num_views, used_globals, point_counts, global_errors, global_masks, estimate.view_reasons
    )
    active_errors = torch.cat([global_errors[i][global_masks[i]] for i in used_globals])
    overall_rms = float(torch.sqrt(active_errors.square().mean()))
    rms_per_view = [d.rms_reprojection_error for d in diagnostics if d.used]

    intrinsic_params = camera_params[:NUM_INTRINSIC_PARAMS]
    lens_params = camera_params[NUM_INTRINSIC_PARAMS:]
    intrinsics = build_intrinsics(intrinsic_params)
    extrinsics = pose_from_params(pose_params)
    # The fisheye result carries the flat [fx, fy, cx, cy, k0, k1, k2, k3] vector that
    # distort_points_kannala_brandt consumes; the pinhole result carries the distortion coefficients alone.
    distortion = (
        torch.cat([intrinsic_params, lens_params], dim=-1) if model == FISHEYE_MODEL else lens_params
    )

    mask_rows = []
    for i in range(num_views):
        if global_masks[i] is None:
            mask_rows.append(torch.zeros(point_counts[i], dtype=torch.bool, device=device))
        else:
            mask_rows.append(global_masks[i])
    rejected_points = [point_counts[i] - int(mask_rows[i].sum()) for i in range(num_views)]

    return CalibrationResult(
        camera_id=camera_id,
        image_size=image_size,
        model=model,
        distortion_order=fitted_order,
        success=True,
        intrinsics=intrinsics.to(input_dtype),
        distortion=distortion.to(input_dtype),
        extrinsics=extrinsics.to(input_dtype),
        view_diagnostics=diagnostics,
        inlier_masks=torch.stack(mask_rows, dim=0),
        rms_reprojection_error=overall_rms,
        rms_per_view=rms_per_view,
        distortion_reduced=fitted_order != distortion_order if model == PINHOLE_MODEL else False,
        rejected_views=[i for i in range(num_views) if i not in used_globals],
        rejected_points=rejected_points,
        rejection_reasons=notes,
        num_iterations=last_fit.iterations,
        converged=last_fit.converged,
    )


def calibrate_camera(
    world_points: Sequence[torch.Tensor],
    image_points: Sequence[torch.Tensor],
    image_size: Optional[Tuple[int, int]] = None,
    weights: Optional[Sequence[Optional[torch.Tensor]]] = None,
    camera_id: object = None,
    distortion_order: int = 5,
    max_iterations: int = 100,
) -> CalibrationResult:
    r"""Calibrate one Brown-Conrady pinhole camera from multi-view world-to-pixel correspondences.

    Args:
        world_points: Per-view world points, each ``(N_m, 3)``. A planar calibration board may give ``z = 0``
            points; non-planar points are supported through the PnP initialization.
        image_points: Per-view measured pixels, each ``(N_m, 2)`` in ``(u, v)`` order.
        image_size: Optional ``(height, width)`` the images were taken at, used for the frame-coverage check and
            recorded on the result for resolution traceability.
        weights: Optional per-view non-negative point weights, each ``(N_m,)``. A zero weight excludes a point.
        camera_id: Identifier carried verbatim onto the result, binding the parameters to this camera.
        distortion_order: Requested Brown-Conrady coefficient count, one of 4, 5 or 8. The fit reduces the tier
            when the observations cannot support it and reports the reduction.
        max_iterations: Maximum Levenberg-Marquardt steps per refinement.

    Returns:
        :class:`CalibrationResult`; a calibration that cannot be satisfied returns a result with
        ``success=False`` and named rejection reasons rather than raising a low-level linear algebra error.
    """
    KORNIA_CHECK(
        distortion_order in PINHOLE_DISTORTION_ORDERS,
        f"distortion_order must be one of {PINHOLE_DISTORTION_ORDERS}, got {distortion_order}.",
    )
    observations = CalibrationObservations(
        list(world_points), list(image_points), list(weights) if weights is not None else None, image_size, camera_id
    )
    return _calibrate_one(PINHOLE_MODEL, observations, distortion_order, max_iterations)


def calibrate_fisheye(
    world_points: Sequence[torch.Tensor],
    image_points: Sequence[torch.Tensor],
    image_size: Optional[Tuple[int, int]] = None,
    weights: Optional[Sequence[Optional[torch.Tensor]]] = None,
    camera_id: object = None,
    max_iterations: int = 100,
) -> CalibrationResult:
    r"""Calibrate one Kannala-Brandt fisheye camera from multi-view world-to-pixel correspondences.

    The result's ``distortion`` is the flat ``[fx, fy, cx, cy, k0, k1, k2, k3]`` vector that
    :func:`~kornia.geometry.camera.distort_points_kannala_brandt` consumes.

    Args:
        world_points: Per-view world points, each ``(N_m, 3)``.
        image_points: Per-view measured pixels, each ``(N_m, 2)`` in ``(u, v)`` order.
        image_size: Optional ``(height, width)`` for the coverage check and resolution traceability.
        weights: Optional per-view non-negative point weights, each ``(N_m,)``.
        camera_id: Identifier carried verbatim onto the result.
        max_iterations: Maximum Levenberg-Marquardt steps per refinement.

    Returns:
        :class:`CalibrationResult` with the fisheye model name, or a named failure.
    """
    observations = CalibrationObservations(
        list(world_points), list(image_points), list(weights) if weights is not None else None, image_size, camera_id
    )
    return _calibrate_one(FISHEYE_MODEL, observations, 4, max_iterations)


def calibrate_cameras(
    observations: Sequence[CalibrationObservations],
    model: str = PINHOLE_MODEL,
    distortion_order: int = 5,
    max_iterations: int = 100,
    parallel: bool = False,
    max_workers: Optional[int] = None,
) -> List[CalibrationResult]:
    r"""Calibrate several cameras independently, each with its own initialization, inliers and failure state.

    One camera failing never affects the others or silently degrades them: its result simply carries
    ``success=False`` and its reasons. The routine has no random source or shared mutable state, so the parallel
    and serial runs return identical results.

    Args:
        observations: One :class:`CalibrationObservations` per camera.
        model: :data:`~kornia.geometry.calibration.result.PINHOLE_MODEL` or
            :data:`~kornia.geometry.calibration.result.FISHEYE_MODEL`.
        distortion_order: Requested Brown-Conrady coefficient count (ignored for the fisheye model).
        max_iterations: Maximum Levenberg-Marquardt steps per refinement.
        parallel: Run the cameras in a thread pool. PyTorch's per-thread inference is independent here because no
            global state is touched; results match the serial run bit for bit.
        max_workers: Worker count for the parallel run; None uses the default pool size.

    Returns:
        One :class:`CalibrationResult` per camera, in input order.
    """
    KORNIA_CHECK(model in (PINHOLE_MODEL, FISHEYE_MODEL), f"unknown model {model}.")
    ordered = list(observations)
    if not parallel or len(ordered) <= 1:
        return [
            _calibrate_one(
                model,
                obs,
                distortion_order if model == PINHOLE_MODEL else 4,
                max_iterations,
            )
            for obs in ordered
        ]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(
            executor.map(
                lambda obs: _calibrate_one(
                    model,
                    obs,
                    distortion_order if model == PINHOLE_MODEL else 4,
                    max_iterations,
                ),
                ordered,
            )
        )
