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

"""Vectorized camera parameterization and projection shared by the calibration optimizers.

The nonlinear refinement differentiates re-projection residuals with respect to flat parameter vectors, so the
two camera models of the public distortion API are re-expressed here as pure functions of such vectors, using the
same forward math as :func:`~kornia.geometry.calibration.distort_points` and
:func:`~kornia.geometry.camera.distort_points_kannala_brandt`:

- the Brown-Conrady pinhole model: 4 intrinsics ``(fx, fy, cx, cy)`` plus ``n`` distortion coefficients;
- the Kannala-Brandt fisheye model: the flat ``[fx, fy, cx, cy, k0, k1, k2, k3]`` vector split into the same four
  intrinsics and four lens coefficients.

Poses use the Rodrigues three-vector (axis-angle) convention. Everything here is deterministic and free of global
state; the functions are plain tensor math so ``torch.func.jacrev`` can differentiate them.
"""

from typing import Callable, Tuple

import torch

from kornia.geometry.calibration.distort import distort_points
from kornia.geometry.camera.distortion_kannala_brandt import distort_points_kannala_brandt
from kornia.geometry.conversions import axis_angle_to_rotation_matrix

#: Number of intrinsic entries (fx, fy, cx, cy) shared by both models.
NUM_INTRINSIC_PARAMS = 4
#: Number of pose entries (Rodrigues vector + translation).
NUM_POSE_PARAMS = 6


def build_intrinsics(intrinsic_params: torch.Tensor) -> torch.Tensor:
    r"""Build a pinhole matrix from the flat ``(fx, fy, cx, cy)`` vector.

    Args:
        intrinsic_params: Tensor with shape ``(..., 4)``.

    Returns:
        Pinhole matrices with shape ``(..., 3, 3)``, zero skew.
    """
    fx, fy, cx, cy = intrinsic_params.unbind(-1)
    zero = torch.zeros_like(fx)
    one = torch.ones_like(fx)
    rows = [
        torch.stack([fx, zero, cx], dim=-1),
        torch.stack([zero, fy, cy], dim=-1),
        torch.stack([zero, zero, one], dim=-1),
    ]
    return torch.stack(rows, dim=-2)


def pose_from_params(pose_params: torch.Tensor) -> torch.Tensor:
    r"""Build a world-to-camera ``[R | t]`` matrix from the six-vector ``(axis-angle, translation)``.

    Args:
        pose_params: Tensor with shape ``(..., 6)``.

    Returns:
        World-to-camera matrices with shape ``(..., 3, 4)``.
    """
    axis_angle = pose_params[..., :3]
    translation = pose_params[..., 3:6]
    rotation = axis_angle_to_rotation_matrix(axis_angle)
    return torch.cat([rotation, translation.unsqueeze(-1)], dim=-1)


def _world_to_camera(world_points: torch.Tensor, pose_params: torch.Tensor) -> torch.Tensor:
    rt = pose_from_params(pose_params)  # (3, 4)
    points_h = torch.cat([world_points, torch.ones_like(world_points[..., :1])], dim=-1)  # (N, 4)
    return points_h @ rt.transpose(-1, -2)  # (N, 3)


def project_pinhole(
    world_points: torch.Tensor,
    intrinsic_params: torch.Tensor,
    distortion: torch.Tensor,
    pose_params: torch.Tensor,
) -> torch.Tensor:
    r"""Project world points through a Brown-Conrady pinhole camera parameterized by flat vectors.

    The projection goes through :func:`~kornia.geometry.calibration.distort_points`, so a calibration fitted with
    this function and later consumed through that function re-projects to the same pixels.

    Args:
        world_points: World points with shape ``(N, 3)``.
        intrinsic_params: ``(fx, fy, cx, cy)`` with shape ``(4,)``.
        distortion: Brown-Conrady coefficients with shape ``(n,)``, ``n`` in 4, 5 or 8.
        pose_params: Rodrigues vector and translation with shape ``(6,)``.

    Returns:
        Pixel coordinates with shape ``(N, 2)``.
    """
    cam_points = _world_to_camera(world_points, pose_params)
    k = build_intrinsics(intrinsic_params)  # (3, 3)
    ideal_pixels = cam_points @ k.transpose(-1, -2)
    ideal_pixels = ideal_pixels[..., :2] / ideal_pixels[..., 2:3]
    return distort_points(ideal_pixels, k, distortion)


def project_fisheye(
    world_points: torch.Tensor,
    intrinsic_params: torch.Tensor,
    lens_params: torch.Tensor,
    pose_params: torch.Tensor,
) -> torch.Tensor:
    r"""Project world points through a Kannala-Brandt fisheye camera parameterized by flat vectors.

    Goes through :func:`~kornia.geometry.camera.distort_points_kannala_brandt`, so downstream consumers of that
    function re-project to the same pixels.

    Args:
        world_points: World points with shape ``(N, 3)``.
        intrinsic_params: ``(fx, fy, cx, cy)`` with shape ``(4,)``.
        lens_params: Fisheye coefficients ``(k0, k1, k2, k3)`` with shape ``(4,)``.
        pose_params: Rodrigues vector and translation with shape ``(6,)``.

    Returns:
        Pixel coordinates with shape ``(N, 2)``.
    """
    cam_points = _world_to_camera(world_points, pose_params)
    normalized = cam_points[..., :2] / cam_points[..., 2:3]  # points on the z = 1 plane
    params = torch.cat([intrinsic_params, lens_params], dim=-1)
    return distort_points_kannala_brandt(normalized, params)


def residual_jacobians(
    project_fn: Callable,
    world_points: torch.Tensor,
    image_points: torch.Tensor,
    camera_params: torch.Tensor,
    pose_params: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Residuals and the two Jacobian blocks of one view via reverse-mode automatic differentiation.

    ``camera_params`` is the shared vector the optimizer holds: intrinsics followed by the distortion
    coefficients. Both projection functions take the same four positional arguments, so the split between the
    intrinsic and distortion parts happens inside them.

    Args:
        project_fn: Either :func:`project_pinhole` or :func:`project_fisheye`.
        world_points: View's world points with shape ``(N, 3)``.
        image_points: Measured pixels with shape ``(N, 2)``.
        camera_params: Shared camera parameter vector with shape ``(P,)``, intrinsics first.
        pose_params: This view's six pose parameters with shape ``(6,)``.

    Returns:
        Tuple ``(residual, j_camera, j_pose)`` with shapes ``(2N,)``, ``(2N, P)`` and ``(2N, 6)``.
    """
    n_intr = NUM_INTRINSIC_PARAMS

    def residuals_flat(cam: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        projected = project_fn(world_points, cam[:n_intr], cam[n_intr:], pose)
        return (projected - image_points).reshape(-1)

    j_cam, j_pose = torch.func.jacrev(residuals_flat, argnums=(0, 1))(camera_params, pose_params)
    residual = residuals_flat(camera_params, pose_params)
    return residual, j_cam, j_pose
