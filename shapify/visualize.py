"""
Compare fitted SMPL shape meshes in an interactive viewer.

The script expects three directories of corresponding `.obj` meshes:
ground truth, method 1, and method 2. It prints pelvis-relative joint errors
and vertex errors, then colors each prediction by per-vertex error.

Usage:
  python -m shapify.visualize --gt_path compare/gt \
      --method_1_path compare/ours_height \
      --method_2_path compare/ours_height_weight
"""
import argparse
import os
import trimesh
import matplotlib.pyplot as plt

import numpy as np


def _viewer_components():
    try:
        from aitviewer.configuration import CONFIG as C
        from aitviewer.viewer import Viewer
        from aitviewer.renderables.meshes import Meshes
        from smplx import SMPL
    except ImportError as exc:
        raise SystemExit("shapify.visualize requires the optional aitviewer dependency.") from exc

    C.window_type = "pyglet"
    return Viewer, Meshes, SMPL


def main(args):
    Viewer, Meshes, SMPL = _viewer_components()

    gt_path = args.gt_path
    method_1_path = args.method_1_path
    method_2_path = args.method_2_path

    # Create the viewer.
    colormap = plt.cm.jet
    viewer_size = None

    viewer = Viewer(size=viewer_size)
    from phd.utils.assets import smpl_model_path
    J_regressor_24_SMPL_neutral = SMPL(model_path=smpl_model_path(),
                                        gender='neutral').J_regressor.cpu().numpy()
    
    gt_meshes = [ x for x in sorted(os.listdir(gt_path)) if x.endswith('.obj') ]
    method_1_meshes = [ x for x in sorted(os.listdir(method_1_path)) if x.endswith('.obj') ]
    method_2_meshes = [ x for x in sorted(os.listdir(method_2_path)) if x.endswith('.obj') ]

    all_v1_mean = []
    all_v1_max = []
    all_v2_mean = []
    all_v2_max = []

    all_j1_mean = []
    all_j1_max = []
    all_j2_mean = []
    all_j2_max = []


    for i, (gt, pred_1, pred_2) in enumerate(zip(gt_meshes, method_1_meshes, method_2_meshes)):
        z_pos = i * -1
        gt_mesh = trimesh.load(os.path.join(gt_path, gt), process=False)
        gt_joints = np.matmul(J_regressor_24_SMPL_neutral, gt_mesh.vertices)

        viewer.scene.add(Meshes( gt_mesh.vertices, gt_mesh.faces, position=[0.0, 0.0, z_pos], name="GT Shape "+ str(i)))

        pred_1_mesh = trimesh.load(os.path.join(method_1_path, pred_1), process=False)

        vertex_error1 = np.sqrt(np.sum((gt_mesh.vertices - pred_1_mesh.vertices) ** 2, axis=1))
        pred_1_joints = np.matmul(J_regressor_24_SMPL_neutral, pred_1_mesh.vertices)
        joint_error1 = np.sqrt(np.sum(((gt_joints-gt_joints[[0]]) - (pred_1_joints-pred_1_joints[[0]])) ** 2, axis=1))



        colors1 = colormap(vertex_error1 * 20)
        viewer.scene.add(Meshes( pred_1_mesh.vertices, pred_1_mesh.faces,
                                 vertex_colors=colors1, position=[2.0, 0.0, z_pos], name="Pred 1 "+ str(i)))

        pred_2_mesh = trimesh.load(os.path.join(method_2_path, pred_2), process=False)

        vertex_error2 = np.sqrt(np.sum((gt_mesh.vertices - pred_2_mesh.vertices) ** 2, axis=1))

        pred_2_joints = np.matmul(J_regressor_24_SMPL_neutral, pred_2_mesh.vertices)
        joint_error2 = np.sqrt(np.sum(((gt_joints-gt_joints[[0]]) - (pred_2_joints-pred_2_joints[[0]])) ** 2, axis=1))

        print(joint_error1.mean(), max(joint_error1), joint_error2.mean(), max(joint_error2), vertex_error1.mean(), max(vertex_error1), vertex_error2.mean(), max(vertex_error2))

        colors2 = colormap(vertex_error2 * 20)
        viewer.scene.add(Meshes( pred_2_mesh.vertices, pred_2_mesh.faces,
                                 vertex_colors=colors2, position=[4.0, 0.0, z_pos], name="Pred 2 "+ str(i)))

        all_v1_mean.append(vertex_error1.mean())
        all_v1_max.append(max(vertex_error1))
        all_v2_mean.append(vertex_error2.mean())
        all_v2_max.append(max(vertex_error2))
        all_j1_mean.append(joint_error1.mean())
        all_j1_max.append(max(joint_error1))
        all_j2_mean.append(joint_error2.mean())
        all_j2_max.append(max(joint_error2))



    print(np.stack(all_j1_mean).mean()*1000,
             np.stack(all_j1_max).mean()*1000,
                np.stack(all_j2_mean).mean()*1000,
                  np.stack(all_j2_max).mean()*1000,
        np.stack(all_v1_mean).mean()*1000,
             np.stack(all_v1_max).mean()*1000,
                np.stack(all_v2_mean).mean()*1000,
                  np.stack(all_v2_max).mean()*1000)

    

    viewer.scene.origin.enabled = True
    viewer.auto_set_floor = False
    viewer.scene.floor.position = np.array([0, -0.3, 0])
    viewer.scene.floor.enabled = False

    viewer.playback_fps = 10.0

    viewer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", default="compare/gt")
    parser.add_argument("--method_1_path", default="compare/ours_height")
    parser.add_argument("--method_2_path", default="compare/ours_height_weight")
    args = parser.parse_args()
    main(args)
