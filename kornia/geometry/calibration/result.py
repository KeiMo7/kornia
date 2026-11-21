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

"""Result and per-view diagnostics of a multi-view camera calibration, and the shared camera model names."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

#: Camera model names a calibration can fit. The pinhole model uses OpenCV's Brown-Conrady coefficient vector and
#: the fisheye model the Kannala-Brandt flat vector, matching :func:`~kornia.geometry.calibration.distort_points`
#: and :func:`~kornia.geometry.camera.distort_points_kannala_brandt`.
PINHOLE_MODEL = "brown_conrady"
FISHEYE_MODEL = "kannala_brandt"

#: The accepted Brown-Conrady distortion coefficient counts, in the order calibration tries to reduce them.
PINHOLE_DISTORTION_ORDERS: Tuple[int, ...] = (8, 5, 4)
#: Kannala-Brandt always carries its four fisheye coefficients on top of the four affine ones.
FISHEYE_NUM_PARAMS = 8


@dataclass
class ViewDiagnostics:
    r"""Per-view diagnostics of a calibration, one entry per input view (including rejected ones).

    All residuals are in pixels and computed by re-projecting the *world* points through the returned camera
    parameters with the same functions the rest of the library consumes
    (:func:`~kornia.geometry.calibration.distort_points` or
    :func:`~kornia.geometry.camera.distort_points_kannala_brandt`), so the numbers here match an independent
    downstream re-projection to rounding.

    Attributes:
        view_index: Index of the view in the input list.
        used: Whether the view contributed to the final fit. ``False`` for views rejected as degenerate or
            gross-outlier views, whose diagnostics are still reported.
        num_points: Number of finite, nonzero-weight points the view supplied.
        num_inliers: Number of those points kept after robust point rejection.
        rms_reprojection_error: Root mean square re-projection error over the inlier points, in pixels.
        max_reprojection_error: Largest inlier re-projection error, in pixels.
        mean_reprojection_error: Mean inlier re-projection error, in pixels.
        rejected: Whether the whole view was dropped from the fit.
        rejection_reason: Human-readable reason when ``rejected`` is True, else None.
        degeneracy: Name of the degeneracy detected on this view, or None. One of ``"fewer_than_four_points"``,
            ``"all_weights_zero"``, ``"non_finite"``, ``"degenerate_homography"`` (the board points do not span a
            plane in the image), ``"small_baseline"`` (the camera barely moved relative to the other views).
    """

    view_index: int
    used: bool
    num_points: int
    num_inliers: int
    rms_reprojection_error: float
    max_reprojection_error: float
    mean_reprojection_error: float
    rejected: bool = False
    rejection_reason: Optional[str] = None
    degeneracy: Optional[str] = None


@dataclass
class CalibrationResult:
    r"""Outcome of one camera's multi-view calibration, bound to a camera id, an image size and a model tier.

    A result is always returned, even on failure: inspect :attr:`success` before consuming the parameters. On
    failure, :attr:`intrinsics`, :attr:`distortion` and :attr:`extrinsics` are ``None`` and :attr:`rejection_reasons`
    names the constraint that could not be satisfied. The binding fields (:attr:`camera_id`, :attr:`image_size`,
    :attr:`model`, :attr:`distortion_order`) are filled on every outcome, so a batch never confuses camera A's
    parameters with camera B's and a result can be traced to the resolution it was fitted at.

    Attributes:
        camera_id: Identifier of the camera this result belongs to; the input ``camera_id`` verbatim.
        image_size: ``(height, width)`` of the images the observations were made at, in pixels.
        model: Camera model name, :data:`PINHOLE_MODEL` or :data:`FISHEYE_MODEL`.
        distortion_order: Number of distortion coefficients of the fitted tier (4, 5 or 8 for the pinhole model;
            4 for the fisheye model's lens coefficients, eight flat parameters including the intrinsics).
        success: Whether a usable parameter set was produced.
        intrinsics: Pinhole matrix ``(3, 3)`` of the camera, or None on failure.
        distortion: Distortion coefficient vector ``(distortion_order,)`` for the pinhole model, or the flat
            Kannala-Brandt ``(8,)`` vector ``[fx, fy, cx, cy, k0, k1, k2, k3]`` for the fisheye model; None on
            failure.
        extrinsics: World-to-camera ``[R | t]`` matrices ``(M, 3, 4)`` of the *used* views, in input-view order;
            None on failure.
        view_diagnostics: One :class:`ViewDiagnostics` per input view, rejected views included.
        inlier_masks: Bool tensor ``(M, N_m)`` marking the points the robust fit kept, one row per input view
            (all False for rejected views); None on failure before any fit.
        rms_reprojection_error: Overall RMS re-projection error over every kept point of every used view, in
            pixels, matching an independent re-projection through the returned parameters.
        rms_per_view: Per-view RMS over the inliers of each used view, in input-view order.
        distortion_reduced: Whether the requested distortion tier was reduced because the observations could not
            support it, with the reason recorded in :attr:`rejection_reasons` as a non-fatal note.
        rejected_views: Indices of views dropped from the final fit.
        rejected_points: Per-view number of points dropped by robust rejection.
        rejection_reasons: Diagnostics: fatal reasons on failure, plus non-fatal notes (distortion reduction,
            view rejection) on success.
        num_iterations: Number of nonlinear iterations the final fit took, or None when no fit ran.
        converged: Whether the nonlinear refinement converged within its iteration budget.
    """

    camera_id: object
    image_size: Optional[Tuple[int, int]]
    model: str
    distortion_order: int
    success: bool = False
    intrinsics: Optional[torch.Tensor] = None
    distortion: Optional[torch.Tensor] = None
    extrinsics: Optional[torch.Tensor] = None
    view_diagnostics: List[ViewDiagnostics] = field(default_factory=list)
    inlier_masks: Optional[torch.Tensor] = None
    rms_reprojection_error: float = float("nan")
    rms_per_view: List[float] = field(default_factory=list)
    distortion_reduced: bool = False
    rejected_views: List[int] = field(default_factory=list)
    rejected_points: List[int] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)
    num_iterations: Optional[int] = None
    converged: bool = False

    @property
    def failed(self) -> bool:
        """Invert :attr:`success` for callers that prefer a failure predicate."""
        return not self.success
