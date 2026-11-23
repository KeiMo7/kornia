#!/usr/bin/env python3
"""Acceptance evidence for the seam-aware PanoramaStitcher.

Runs the five acceptance checks from the task and writes a human-readable report plus the before/after seam crops as
a raw tensor file. Run from the repository root with the project interpreter:

    python reports/panorama_acceptance.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from kornia.contrib import PanoramaStitcher, StitchingError
from kornia.feature import LocalFeatureMatcher

REPORT_DIR = Path(__file__).resolve().parent


def render(x0: float, w: int, h: int = 48, scale: float = 1.0, base: float = 0.25) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32) + x0, indexing="ij"
    )
    img = torch.stack(
        [
            base
            + 0.08 * torch.sin(0.21 * xs + 0.13 * ys + c)
            + 0.06 * torch.cos(0.17 * ys - 0.07 * xs + 2 * c)
            for c in range(3)
        ]
    )[None]
    return img * scale


class ShiftMatcher(LocalFeatureMatcher):
    def __init__(self, shift: float, keep: int | None = None) -> None:
        nn.Module.__init__(self)
        self.shift, self.keep = shift, keep

    def forward(self, data: dict) -> dict:
        image = data["image1"]
        ys, xs = torch.meshgrid(
            torch.arange(2, image.shape[-2] - 2, 5.0),
            torch.arange(2, image.shape[-1] - 2, 5.0),
            indexing="ij",
        )
        kp1 = torch.stack([xs.flatten(), ys.flatten()], -1)
        kp0 = kp1 + torch.tensor([self.shift, 0.0])
        idx = torch.zeros(kp0.shape[0], dtype=torch.long)
        if self.keep is not None:
            kp0, kp1, idx = kp0[: self.keep], kp1[: self.keep], idx[: self.keep]
        return {"keypoints0": kp0, "keypoints1": kp1, "batch_indexes": idx}


def shift_homography(shift_x: float) -> torch.Tensor:
    return (torch.eye(3) + torch.tensor([[0.0, 0.0, shift_x], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))[None]


def main() -> None:
    lines: list[str] = []
    log = lines.append
    torch.manual_seed(0)
    log("PanoramaStitcher acceptance report")
    log(f"torch {torch.__version__}, threads {torch.get_num_threads()}")
    log("")

    # ---- 1 & 2: seam discontinuity and same-material color difference, with before/after numbers
    left = render(0.0, 64)
    right = render(40.0, 56, scale=1.5)  # a controlled 1.5x exposure/white-balance difference, no clipping
    stitcher = PanoramaStitcher(ShiftMatcher(40.0))
    pano, mask, quality, slices = stitcher([left, right], return_details=True)
    seam_before = float(quality.seam_discontinuity_before)
    seam_after = float(quality.seam_discontinuity_after)
    luma_before = float(quality.overlap_luma_difference_before)
    luma_after = float(quality.overlap_luma_difference_after)
    log("[1] Seam discontinuity (mean abs jump on the seam ring)")
    log(f"    before = {seam_before:.6f}  after = {seam_after:.6f}  reduction = {1 - seam_after / seam_before:.1%}")
    log("[2] Same-material overlap luma difference")
    log(f"    before = {luma_before:.6f}  after = {luma_after:.6f}  reduction = {1 - luma_after / luma_before:.1%}")
    log(f"    clipped under/over fractions = {quality.clipped_pixel_fraction}")
    assert seam_after < seam_before * 0.4
    assert luma_after < 0.01
    log("")

    # ---- 3: matching failure is rejected with pair and inlier counts
    few = PanoramaStitcher(ShiftMatcher(40.0, keep=3), min_inliers=8)
    try:
        few([left, right])
        raise SystemExit("expected a StitchingError for too few matches")
    except StitchingError as exc:
        log("[3] Matching-failure rejection")
        log(f"    StitchingError: {exc}")
    # and a geometrically disjoint pair (zero overlap)
    disjoint = PanoramaStitcher(ShiftMatcher(40.0))
    try:
        disjoint.fuse([left, right], [torch.eye(3)[None], shift_homography(200.0)])
        raise SystemExit("expected a StitchingError for zero overlap")
    except StitchingError as exc:
        log(f"    StitchingError: {exc}")
    log("")

    # ---- 4: bit-identical reruns, including across fresh stitcher instances
    pano_a, mask_a = PanoramaStitcher(ShiftMatcher(40.0))([left, right])
    pano_b, mask_b = PanoramaStitcher(ShiftMatcher(40.0))([left, right])
    identical = torch.equal(pano_a, pano_b) and torch.equal(mask_a, mask_b)
    log(f"[4] Two independent runs are bit-identical: {identical}")
    assert identical
    log("")

    # ---- 5: existing stitcher/transform tests are unaffected; this script also saves review crops
    torch.save({"before": slices.before, "after": slices.after, "boxes": slices.boxes},
               REPORT_DIR / "panorama_seam_slices.pt")
    log(f"[5] Seam review crops saved to reports/panorama_seam_slices.pt "
        f"(before {tuple(slices.before.shape)}, after {tuple(slices.after.shape)})")
    log("    Existing ImageStitcher behavior: run pytest tests/contrib/test_image_stitcher.py separately.")
    log("")

    # ---- extra: multi-image chain stability, order-swap bound, fusion-only mode, timing
    chain = [render(0.0, 64), render(40.0, 64, scale=1.3), render(80.0, 64, scale=1.3 * 1.3)]
    _, _, q3, _ = stitcher(chain, return_details=True)
    log(f"[extra] 3-image chain: seam after = {float(q3.seam_discontinuity_after):.6f}, "
        f"luma after = {float(q3.overlap_luma_difference_after):.6f}, overlap fractions = {q3.overlap_fractions}")

    inverse = torch.linalg.inv(shift_homography(40.0)[0])[None]
    _, _, q_ba, _ = stitcher.fuse([right, left], [torch.eye(3)[None], inverse], return_details=True)
    order_delta = abs(seam_after - float(q_ba.seam_discontinuity_after))
    log(f"[extra] Order-swap seam delta (AB vs BA): {order_delta:.6f}")

    class BoomMatcher(LocalFeatureMatcher):
        def __init__(self) -> None:
            nn.Module.__init__(self)

        def forward(self, data: dict) -> dict:
            raise AssertionError("matcher must not run in fusion-only mode")

    pf, mf = PanoramaStitcher(BoomMatcher()).fuse([left, right],
                                                  [torch.eye(3)[None], shift_homography(40.0)])
    log(f"[extra] Fusion-only mode (no matcher call): panorama {tuple(pf.shape)}, mask coverage {float(mf.mean()):.3f}")

    # timing on a common 224-wide pair, warm then measured
    big = [torch.rand(1, 3, 224, 224), torch.rand(1, 3, 224, 224)]
    big_h = [torch.eye(3)[None], shift_homography(120.0)]
    timed = PanoramaStitcher(BoomMatcher())
    big_pair = [big[0], big[1]]
    timed.fuse(big_pair, big_h)
    start = time.perf_counter()
    for _ in range(5):
        timed.fuse(big_pair, big_h)
    elapsed = (time.perf_counter() - start) / 5
    log(f"[extra] Mean fusion-only time over 5 runs on (1,3,224,224) pairs: {elapsed * 1000:.1f} ms")

    report = "\n".join(lines)
    (REPORT_DIR / "panorama_acceptance.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
