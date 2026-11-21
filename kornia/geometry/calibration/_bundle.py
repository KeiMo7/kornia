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

"""Levenberg-Marquardt bundle adjustment shared by the pinhole and fisheye calibrations.

The normal equations are accumulated one view at a time and the six pose parameters of every view are eliminated
with the Schur complement, so only the small shared-camera block is solved directly. Robustness comes from
iteratively reweighted least squares with a Huber kernel whose scale is the median absolute deviation of the
re-projection residuals: the whole routine is deterministic, with no random sampling and no global state.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from kornia.geometry.calibration._projection import (
    NUM_INTRINSIC_PARAMS,
    residual_jacobians,
)

# Damping parameters of the Levenberg-Marquardt trust region. The factor multiplies the diagonal of the normal
# equations (Marquardt's relative scaling), so the first step is a short gradient-like step even when the
# intrinsics guess is far off, and a successful step moves the factor toward pure Gauss-Newton.
_INITIAL_DAMPING = 1e-2
_DAMPING_UP = 10.0
_DAMPING_DOWN = 10.0
_MAX_DAMPING = 1e12
_MIN_DAMPING = 1e-10
# Huber's 1.345 gives 95% efficiency under Gaussian residuals.
_HUBER_C = 1.345
# Minimum robust scale in pixels; below this the data are treated as effectively exact and nothing is rejected.
_MIN_ROBUST_SCALE = 1e-3
_EPS = 1e-12


@dataclass
class LMResult:
    r"""Outcome of one Levenberg-Marquardt run.

    Attributes:
        camera_params: Fitted shared camera parameter vector.
        pose_params: Fitted per-view pose vectors, shape ``(M, 6)``.
        iterations: Number of accepted steps.
        converged: Whether the step reached the gradient/parameter tolerance before the iteration budget ran out.
        finite: Whether every quantity of the final state is finite.
        point_residuals: Per-point Euclidean re-projection error of every view, in pixels.
    """

    camera_params: torch.Tensor
    pose_params: torch.Tensor
    iterations: int
    converged: bool
    finite: bool
    point_residuals: List[torch.Tensor]


def _robust_weights_and_scale(
    point_errors: List[torch.Tensor], inlier_masks: Sequence[torch.Tensor]
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Huber IRLS weights from the MAD scale of the inlier point errors, one weight vector per view."""
    active = torch.cat([e[m] for e, m in zip(point_errors, inlier_masks) if bool(m.any())])
    if active.numel() == 0:
        scale = torch.tensor(_MIN_ROBUST_SCALE, dtype=point_errors[0].dtype, device=point_errors[0].device)
    else:
        median = torch.median(active)
        mad = torch.median(torch.abs(active - median))
        # 1.4826 makes MAD a consistent standard-deviation estimate under Gaussian noise.
        scale = torch.clamp(1.4826 * mad, min=_MIN_ROBUST_SCALE)
    weights = []
    for errors, mask in zip(point_errors, inlier_masks):
        w = torch.ones_like(errors)
        huber = _HUBER_C * scale
        large = errors > huber
        w[large] = huber / errors[large].clamp_min(_EPS)
        weights.append(w * mask.to(w.dtype))
    return weights, scale


def _evaluate(
    project_fn: Callable,
    views: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    camera_params: torch.Tensor,
    pose_params: torch.Tensor,
    inlier_masks: Sequence[torch.Tensor],
    weights: Optional[Sequence[torch.Tensor]],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Weighted sum of squared re-projection residuals and the per-point Euclidean errors."""
    cost = camera_params.new_zeros(())
    point_errors: List[torch.Tensor] = []
    for i, (world_points, image_points) in enumerate(views):
        projected = project_fn(
            world_points,
            camera_params[:NUM_INTRINSIC_PARAMS],
            camera_params[NUM_INTRINSIC_PARAMS:],
            pose_params[i],
        )
        errors = torch.linalg.vector_norm(projected - image_points, dim=-1)
        point_errors.append(errors)
        w = weights[i] if weights is not None else inlier_masks[i].to(errors.dtype)
        cost = cost + (w * errors.square()).sum()
    return cost, point_errors


def levenberg_marquardt(
    project_fn: Callable,
    views: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    camera_params: torch.Tensor,
    pose_params: torch.Tensor,
    inlier_masks: Sequence[torch.Tensor],
    max_iterations: int = 100,
    tol: float = 1e-8,
    robust: bool = True,
    camera_active_mask: Optional[torch.Tensor] = None,
) -> LMResult:
    r"""Refine shared camera parameters and one pose per view against re-projection residuals.

    Args:
        project_fn: :func:`~kornia.geometry.calibration._projection.project_pinhole` or
            :func:`~kornia.geometry.calibration._projection.project_fisheye`.
        views: ``M`` tuples of world points ``(N_m, 3)`` and measured pixels ``(N_m, 2)``.
        camera_params: Initial shared camera vector (intrinsics then distortion), shape ``(P,)``.
        pose_params: Initial per-view pose vectors, shape ``(M, 6)``.
        inlier_masks: Bool tensors ``(N_m,)`` marking the points each view contributes.
        max_iterations: Maximum number of accepted LM steps.
        tol: Convergence tolerance on the relative cost decrease and on the gradient norm.
        robust: Recompute Huber IRLS weights from the residual MAD after every accepted step. When False, every
            inlier point keeps unit weight.
        camera_active_mask: Optional bool tensor ``(P,)``; a False entry freezes that camera parameter (its step is
            forced to zero), used to fit intrinsics and poses with zero distortion before the distortion tier is
            released. All parameters are active when None.

    Returns:
        :class:`LMResult` with the fitted parameters and the final per-point errors.
    """
    dtype, device = camera_params.dtype, camera_params.device
    num_views = len(views)
    num_cam = camera_params.shape[0]
    damping = _INITIAL_DAMPING
    weights: Optional[List[torch.Tensor]] = None

    # Jacobi preconditioning on the shared camera parameters. Focal lengths and the principal point are O(10^2-10^3)
    # while distortion coefficients are O(10^-1); without rescaling, the damping ignores the distortion block and
    # the fit stalls. Each camera parameter is optimized in units of its own magnitude (floored for near-zero
    # entries), and the step is converted back before being applied.
    camera_scale = camera_params.detach().abs().clamp_min(1.0)
    if camera_active_mask is None:
        camera_active_mask = torch.ones(num_cam, dtype=torch.bool, device=device)
    inactive = (~camera_active_mask).to(dtype)

    cost, point_errors = _evaluate(project_fn, views, camera_params, pose_params, inlier_masks, weights)
    if robust:
        weights, _ = _robust_weights_and_scale(point_errors, inlier_masks)
        cost, point_errors = _evaluate(project_fn, views, camera_params, pose_params, inlier_masks, weights)

    iterations = 0
    converged = False
    for _ in range(max_iterations):
        # Accumulate the undamped normal equations view by view, then Schur-eliminate each pose block.
        h_cam = torch.zeros((num_cam, num_cam), dtype=dtype, device=device)
        g_cam = torch.zeros(num_cam, dtype=dtype, device=device)
        schur_rhs_pose: List[torch.Tensor] = []
        schur_blocks: List[torch.Tensor] = []
        pose_jacobians: List[torch.Tensor] = []
        for i, (world_points, image_points) in enumerate(views):
            mask = inlier_masks[i]
            if not bool(mask.any()):
                schur_blocks.append(None)
                schur_rhs_pose.append(None)
                pose_jacobians.append(None)
                continue
            residual, j_cam, j_pose = residual_jacobians(
                project_fn, world_points[mask], image_points[mask], camera_params, pose_params[i]
            )
            if weights is not None:
                w = weights[i][mask]
            else:
                w = torch.ones(residual.shape[0] // 2, dtype=dtype, device=device)
            w2 = w.repeat_interleave(2).unsqueeze(-1)  # one weight per point, applied to its two residual rows
            j_cam = j_cam * w2 * camera_scale  # derivative w.r.t. the scaled camera parameters
            j_cam = j_cam * camera_active_mask.to(dtype)  # frozen parameters get no column, hence no step
            j_pose = j_pose * w2
            residual_w = residual * w2.squeeze(-1)

            a = j_pose.transpose(-1, -2) @ j_pose  # (6, 6)
            b = j_pose.transpose(-1, -2) @ j_cam  # (6, P)
            g_pose = j_pose.transpose(-1, -2) @ residual_w  # (6,)
            # Marquardt damping on the pose block: relative to its own diagonal, floored so a zero entry stays usable.
            a_damped = a + damping * torch.diag(torch.diagonal(a).clamp_min(_EPS))
            a_inv = torch.linalg.inv(a_damped)
            # Normal equations in (delta_pose, delta_cam), with g = J^T r so that H delta = g and theta += delta.
            # Schur complement on the camera block: H_cc -= B^T A^-1 B, g_c -= B^T A^-1 g_pose.
            h_cam = h_cam + j_cam.transpose(-1, -2) @ j_cam - b.transpose(-1, -2) @ a_inv @ b
            g_cam = g_cam + j_cam.transpose(-1, -2) @ residual_w - b.transpose(-1, -2) @ a_inv @ g_pose
            schur_blocks.append(a_inv)
            schur_rhs_pose.append(g_pose)
            pose_jacobians.append(b)

        h_damped = h_cam + damping * torch.diag(torch.diagonal(h_cam).clamp_min(_EPS))
        # Frozen columns are zero, which would make the damped block singular; give those rows a unit penalty so
        # the solve stays well-posed, then force their step to zero.
        h_damped = h_damped + torch.diag(inactive)
        try:
            # Gradient of the squared cost is g = J^T r, so the descent step solves H delta = g and subtracts it.
            delta_cam = torch.linalg.solve(h_damped, g_cam)
        except RuntimeError:
            damping = min(damping * _DAMPING_UP, _MAX_DAMPING)
            if damping >= _MAX_DAMPING:
                break
            continue
        if not bool(torch.isfinite(delta_cam).all()):
            damping = min(damping * _DAMPING_UP, _MAX_DAMPING)
            if damping >= _MAX_DAMPING:
                break
            continue
        delta_cam = delta_cam * camera_active_mask.to(dtype)  # frozen parameters do not move

        # Back-substitute the pose steps from the eliminated blocks: A delta_p = g_p - B delta_c.
        new_pose_params = pose_params.clone()
        for i in range(num_views):
            if schur_blocks[i] is None:
                continue
            delta_pose = schur_blocks[i] @ (
                schur_rhs_pose[i] - pose_jacobians[i] @ delta_cam
            )
            new_pose_params[i] = pose_params[i] - delta_pose
        # The camera step was solved in scaled units; convert it back to physical units before subtracting.
        new_camera_params = camera_params - delta_cam * camera_scale

        new_cost, new_errors = _evaluate(
            project_fn, views, new_camera_params, new_pose_params, inlier_masks, weights
        )
        if bool(torch.isfinite(new_cost).all()) and new_cost < cost:
            relative_decrease = (cost - new_cost) / cost.clamp_min(_EPS)
            camera_params = new_camera_params
            pose_params = new_pose_params
            point_errors = new_errors
            iterations += 1
            damping = max(damping / _DAMPING_DOWN, _MIN_DAMPING)
            # Scale-free stopping: either the step no longer changes the cost meaningfully, or the largest
            # gradient component (in squared-cost units per parameter) is at the working precision.
            grad_inf = float(g_cam.abs().max())
            if relative_decrease < tol or grad_inf < tol:
                converged = True
                cost = new_cost
                break
            cost = new_cost
            if robust:
                weights, _ = _robust_weights_and_scale(point_errors, inlier_masks)
                cost, point_errors = _evaluate(
                    project_fn, views, camera_params, pose_params, inlier_masks, weights
                )
        else:
            damping = min(damping * _DAMPING_UP, _MAX_DAMPING)
            if damping >= _MAX_DAMPING:
                break

    finite = bool(torch.isfinite(camera_params).all()) and all(
        bool(torch.isfinite(p).all()) for p in pose_params
    )
    return LMResult(camera_params, pose_params, iterations, converged, finite, point_errors)
