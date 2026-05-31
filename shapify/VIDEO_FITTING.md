# SHAPify Video — Multi-view shape fitting

Goal: recover a subject's SMPL shape (β) **without** requiring a T-pose. The
input is a short smartphone video where the **subject is static** and the
**camera moves** (e.g. the user walks around them).

## Why we can't reuse the single-image fitter

`shapify.fit_shape` is built around three structural assumptions:

1. **Subject faces the camera.** The optimizer parameterizes the body's
   orientation in the camera frame as axis-angle pitch / yaw / roll, where
   pitch ≈ ±π (pelvis-to-camera) is the basin of convergence.
2. **One image, one body, one camera.** The "global orientation" and "camera
   translation" are jointly the only spatial DoF.
3. **T-pose silhouette resolves shape ambiguity.** Shoulder-width pixel
   distance is a depth cue; mesh height + mass anchor scale.

In the moving-camera video setting:

- The subject can be in **any natural static pose**. A pelvis-to-camera
  orientation per frame is not just unhelpful — splitting that into
  pitch/yaw/roll and reusing the same per-frame parameterization causes the
  body to **twist independently in each view to overfit 2D keypoints**.
  N copies of (orient, body_pose) describing "the same body seen from N
  angles" is N times more parameters than the problem needs.
- There is no single camera; there are N cameras tracing a trajectory. The
  important geometric content is the **relative pose between cameras**, not
  a per-frame body orientation.
- A single 2D silhouette is no longer the only shape cue: **multi-view
  consistency replaces the T-pose constraint**.

## Variables

We separate **body parameters** (shared across all frames; the subject is
static) from **camera parameters** (one set per frame; the camera moves).

### Body (shared across frames)

| Symbol | Shape | Meaning |
|---|---|---|
| `β` | 10 | SMPL shape coefficients |
| `θ` | 23 × 6 (rot6d) | SMPL body pose |
| `R_body→cam₀` | 3 × 3 (stored as 6) | Pelvis-to-cam₀ orientation |
| `T_body in cam₀` | 3 | Position of pelvis in cam₀'s frame |

### Camera (per-frame, i = 1 … N−1)

| Symbol | Shape | Meaning |
|---|---|---|
| `R_camᵢ←cam₀` | 3 × 3 (stored as 6) | Rotation of cam_i relative to cam_0 |
| `T_camᵢ←cam₀` | 3 | Translation of cam_i relative to cam_0 |

**Cam_0 is the world** by convention: `R_cam₀←cam₀ = I`, `T_cam₀←cam₀ = 0`.
Only cameras 1 … N−1 are optimized as relative transforms. This collapses the
gauge freedom (any global rotation/translation of the body + cameras is
unobservable) by anchoring everything to cam₀.

### Why 6D rotation, not axis-angle

Axis-angle is double-covered: a rotation θ around axis n is identical to
−(2π − θ) around the same axis. The PointDiT-init "pelvis-to-camera" orient
sits near ±π, exactly where axis-angle is singular and the standard
quaternion→axis-angle helper silently flips signs (verified locally — `R(−π)`
round-trips to `+π`). 6D rotation representation (Zhou et al. 2019) is
continuous and surjective onto SO(3), so the optimizer doesn't fight
double-cover discontinuities.

## Forward model

For frame i, the body's vertices in cam_i frame are:

```
J_body_canonical  = SMPL(global_orient = I,  body_pose = θ,  betas = β).joints
                                                                # body-canonical frame
J_body_centered   = J_body_canonical − J_body_canonical[pelvis]

R_camᵢ←body       = R_camᵢ←cam₀ · R_body→cam₀
T_camᵢ←body       = R_camᵢ←cam₀ · T_body_in_cam₀ + T_camᵢ←cam₀

J_camᵢ            = R_camᵢ←body · J_body_centered + T_camᵢ←body
proj_2dᵢ          = π_Kᵢ(J_camᵢ)
```

Where `π_Kᵢ` is perspective projection with frame-i intrinsics.

This formulation enforces that **the same body** (β, θ) is being viewed; the
per-frame variation is purely the camera's SE(3) transform from cam₀.

## Losses

| Term | Why |
|---|---|
| 2D keypoint reprojection — OpenPose 25 body joints, per frame, weighted by detection confidence | Pulls projected joints toward detected ones in every view |
| `|mesh_height(β) − target_height|` | Anchors metric scale. Uses pose-zero canonical mesh (pose-agnostic). |
| `|mesh_mass(β) − target_weight|` (volume × 985 kg/m³) | Disambiguates β scale orthogonally to height |
| `‖β − β_prior‖²` toward gender prior | Stabilizes β when measurements are noisy |
| `‖θ − θ_init‖²` toward PointDiT-averaged init | **Critical**: prevents the body from twisting to fit 2D keypoints. The kp loss alone is insufficient because 25 sparse 2D joints don't fully constrain 69 DoF of body pose; the regularizer keeps the body near a 3D-plausible PointDiT prior. |
| (optional) camera smoothness | Smartphone trajectories are smooth |

## Initialization

PointDiT runs once per frame (still needed for the body params); the
**camera trajectory** is then recovered by robust PnP rather than by
composing PointDiT's noisy per-frame rotations.

1. **Frame 0 anchors the body.** PointDiT (single frame, the closest-to-front
   one by default = first frame in input order) gives:
   - `R_body→cam₀` — pelvis-to-camera rotation
   - `T_body_in_cam₀` — pelvis position in cam_0 frame (via
     `phd.camera.find_cam_pos` — a weighted linear-LS solve from frame-0
     OpenPose + PointDiT's 3D joints)
   - `θ_init` (body_pose) — used as the shared body_pose init **and** to
     build the 3D anchors for PnP (consistent pair)
   - β_init — gender prior

2. **3D body anchors in cam_0 frame.** Run SMPL forward once with the
   frame-0 params; index `joints[SMPL_TO_OPENPOSE]` → 25 metric 3D points
   anchored in cam_0's world.

3. **Per-frame PnP, occlusion-aware.** For each frame i > 0:
   - Filter OpenPose-25 detections by confidence (default 0.3); low-conf
     joints are treated as occluded and skipped.
   - Need ≥ 6 confident joints; otherwise **drop the frame entirely** from
     the multi-view fit.
   - `cv2.solvePnPRansac(3D=joints_op25_cam0[mask], 2D=keypoints_op25[mask], K=K_i)`
     with `SOLVEPNP_EPNP` and RANSAC outlier rejection (reproj-err threshold
     8 px). RANSAC inliers refine the PnP solve.
   - cv2 returns rvec / tvec in **world → cam** convention, which matches
     `R_camᵢ←cam₀` / `T_camᵢ←cam₀` directly. No conversion needed.

4. **Why not the older "transitive composition" init?** Composing
   `R_body→camᵢ · R_body→cam₀ᵀ` makes every per-frame camera init depend
   on PointDiT's rotation accuracy for frame i — which on synthetic /
   stylized renders is noisy enough to bias the optimizer into a bad mode.
   PnP from cam_0's 3D joints uses **only** OpenPose-2D for frames i > 0
   (PointDiT is no longer on the critical path for cam_i ≠ 0).

## Outputs

- `neutral_shape<id>.npy` — 10-D β (drop-in for the body fitter).
- `pred_shape<id>.obj` — canonical SMPL mesh in T-pose with recovered β.
- `opt_mesh_<id>.obj` — frame-0 posed mesh in cam₀ frame.
- `body_pose_rotmat<id>.npy`, `R_body_to_cam0<id>.npy`, `T_body_in_cam0<id>.npy` — recovered body params.
- `R_cam_i_from_cam0<id>.npy`, `T_cam_i_from_cam0<id>.npy` — recovered camera trajectory (in cam₀-relative coordinates).
- `opt_frame_NN_<id>.jpg` — per-frame mesh overlay onto the input image.

## Known limitations

- **Body pose recovery is approximate.** With only 25 OpenPose joints per
  frame and the multi-view consistency constraint, body_pose can still drift
  10-20° at individual joints without being heavily penalized by 2D loss.
  This is acceptable for **shape recovery** (the goal here) but not for
  motion capture-grade pose. Stronger pose priors (VPoser, PointDiT-3D
  consistency across iterations) would be the next improvement.
- **Scale anchor still requires measurements.** Without height + weight, β
  scale is ambiguous unless cameras are metric (ARKit / IMU / known
  baseline). Dropping measurements would require a metric camera source —
  not implemented yet.
- **PointDiT-domain mismatch.** PointDiT was trained on BEDLAM photoreal
  renders. Synthetic gray-background renders (or stylized captures) can
  yield noisy per-frame estimates; the multi-view averaging and tight pose
  regularizer compensate, but real smartphone capture should perform better.
