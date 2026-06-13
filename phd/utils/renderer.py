import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import numpy as np
import pyrender
import trimesh
import cv2
from typing import List, Optional

def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.):
    # Convert cam_bbox to full image
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam

def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses

def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)

def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))

def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)


def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )


def _axis_angle_rotation_matrix(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float32)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        return np.eye(3, dtype=np.float32)

    x, y, z = axis / axis_norm
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)
    one_minus_cos = 1.0 - cos_t

    return np.array(
        [
            [cos_t + x * x * one_minus_cos, x * y * one_minus_cos - z * sin_t, x * z * one_minus_cos + y * sin_t],
            [y * x * one_minus_cos + z * sin_t, cos_t + y * y * one_minus_cos, y * z * one_minus_cos - x * sin_t],
            [z * x * one_minus_cos - y * sin_t, z * y * one_minus_cos + x * sin_t, cos_t + z * z * one_minus_cos],
        ],
        dtype=np.float32,
    )
    

def create_raymond_lights() -> List[pyrender.Node]:
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes

class Renderer:

    def __init__(self, faces: np.array, backend: str = "auto"):
        """
        Wrapper around the mesh renderer used to render SMPL meshes.
        Args:
            faces (np.array): Array of shape (F, 3) containing the mesh faces.
            backend: "auto", "kaolin", or "pyrender". "auto" uses Kaolin on
                CUDA when available and falls back to pyrender otherwise.
        """
        self.focal_length = 5000
        self.img_res = 256

        self.camera_center = [self.img_res // 2, self.img_res // 2]
        self.faces = np.asarray(faces, dtype=np.int64)
        self.backend = backend
        self._kaolin_mesh = None

    def __call__(self,
                vertices: np.array,
                camera_translation: np.array,
                image: torch.Tensor,
                full_frame: bool = False,
                imgname: Optional[str] = None,
                side_view=False, rot_angle=90,
                mesh_base_color=(1.0, 1.0, 0.9),
                scene_bg_color=(0,0,0),
                return_rgba=False,
                ) -> np.array:
        """
        Render meshes on input image
        Args:
            vertices (np.array): Array of shape (V, 3) containing the mesh vertices.
            camera_translation (np.array): Array of shape (3,) with the camera translation.
            image (torch.Tensor): Tensor of shape (3, H, W) containing the image crop with normalized pixel values.
            full_frame (bool): If True, then render on the full image.
            imgname (Optional[str]): Contains the original image filenamee. Used only if full_frame == True.
        """
        
        if full_frame:
            image = cv2.imread(imgname).astype(np.float32)[:, :, ::-1] / 255.
        else:
            image = image.clone() * torch.tensor([0.229, 0.224, 0.225], device=image.device).reshape(3,1,1)
            image = image + torch.tensor([0.485, 0.456, 0.406], device=image.device).reshape(3,1,1)
            image = image.permute(1, 2, 0).cpu().numpy()

        renderer = pyrender.OffscreenRenderer(viewport_width=image.shape[1],
                                              viewport_height=image.shape[0],
                                              point_size=1.0)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(*mesh_base_color, 1.0))

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        if side_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [0, 1, 0])
            mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [image.shape[1] / 2., image.shape[0] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)
        scene.add(camera, pose=camera_pose)


        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        if return_rgba:
            return color

        valid_mask = (color[:, :, -1])[:, :, np.newaxis]
        if not side_view:
            output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)
        else:
            output_img = color[:, :, :3]

        output_img = output_img.astype(np.float32)
        return output_img

    def vertices_to_trimesh(self, vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9), 
                            rot_axis=[1,0,0], rot_angle=0,):
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))
        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        # print(vertices.shape, camera_translation.shape)
        mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces.copy(), vertex_colors=vertex_colors)
        # mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        
        rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

    def render_rgba(
            self,
            vertices: np.array,
            cam_t = None,
            rot=None,
            rot_axis=[1,0,0],
            rot_angle=0,
            camera_z=3,
            # camera_translation: np.array,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0,0,0),
            render_res=[256, 256],
            focal_length = None
        ):
        if self._use_kaolin():
            return self._render_rgba_kaolin(
                vertices,
                cam_t=cam_t,
                rot_axis=rot_axis,
                rot_angle=rot_angle,
                camera_z=camera_z,
                mesh_base_color=mesh_base_color,
                scene_bg_color=scene_bg_color,
                render_res=render_res,
                focal_length=focal_length,
            )
        return self._render_rgba_pyrender(
            vertices,
            cam_t=cam_t,
            rot_axis=rot_axis,
            rot_angle=rot_angle,
            camera_z=camera_z,
            mesh_base_color=mesh_base_color,
            scene_bg_color=scene_bg_color,
            render_res=render_res,
            focal_length=focal_length,
        )

    def _use_kaolin(self):
        if self.backend == "pyrender":
            return False
        if self.backend not in {"auto", "kaolin"}:
            raise ValueError(f"Unknown renderer backend {self.backend!r}")
        if not torch.cuda.is_available():
            if self.backend == "kaolin":
                raise RuntimeError("Kaolin renderer requires CUDA.")
            return False
        if self._kaolin_mesh is None:
            try:
                from kaolin.render import mesh as kaolin_mesh
            except Exception:
                if self.backend == "kaolin":
                    raise
                return False
            self._kaolin_mesh = kaolin_mesh
        return True

    def _render_rgba_pyrender(
            self,
            vertices: np.array,
            cam_t=None,
            rot_axis=[1, 0, 0],
            rot_angle=0,
            camera_z=3,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0, 0, 0),
            render_res=[256, 256],
            focal_length=None,
        ):
        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))

        if cam_t is not None:
            camera_translation = cam_t.copy()
            # camera_translation[0] *= -1.
        else:
            camera_translation = np.array([0, 0, camera_z * self.focal_length/render_res[1]])

        mesh = self.vertices_to_trimesh(vertices, camera_translation, mesh_base_color, rot_axis, rot_angle)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        # mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        # camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        focal_length = focal_length if focal_length is not None else self.focal_length

        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color

    def _render_rgba_kaolin(
            self,
            vertices: np.array,
            cam_t=None,
            rot_axis=[1, 0, 0],
            rot_angle=0,
            camera_z=3,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0, 0, 0),
            render_res=[256, 256],
            focal_length=None,
        ):
        width, height = int(render_res[0]), int(render_res[1])
        focal_length = float(focal_length if focal_length is not None else self.focal_length)
        device = torch.device("cuda")
        dtype = torch.float32

        vertices_np = np.asarray(vertices, dtype=np.float32).copy()
        if cam_t is not None:
            camera_translation = np.asarray(cam_t, dtype=np.float32).reshape(3)
        else:
            camera_translation = np.array([0.0, 0.0, camera_z * focal_length / height], dtype=np.float32)
        vertices_np = vertices_np + camera_translation.reshape(1, 3)

        if rot_angle:
            rot_mat = _axis_angle_rotation_matrix(rot_axis, np.radians(rot_angle)).astype(np.float32)
            vertices_np = vertices_np @ rot_mat.T

        # Match pyrender's OpenGL camera convention after Renderer.vertices_to_trimesh():
        # a 180 degree X rotation puts visible geometry in front of a camera looking -Z.
        vertices_np = vertices_np * np.array([1.0, -1.0, -1.0], dtype=np.float32).reshape(1, 3)

        vertices_t = torch.from_numpy(vertices_np).to(device=device, dtype=dtype).unsqueeze(0)
        faces_t = torch.from_numpy(self.faces).to(device=device, dtype=torch.long)

        depth = -vertices_t[..., 2]
        x_pix = focal_length * vertices_t[..., 0] / depth.clamp(min=1e-4) + width / 2.0
        y_pix = focal_length * vertices_t[..., 1] / depth.clamp(min=1e-4) + height / 2.0
        x_ndc = x_pix / (width - 1.0) * 2.0 - 1.0
        y_ndc = y_pix / (height - 1.0) * 2.0 - 1.0
        vertices_image = torch.stack([x_ndc, y_ndc], dim=-1)

        face_vertices_camera = vertices_t[:, faces_t]
        face_vertices_image = vertices_image[:, faces_t]
        # Kaolin's CUDA rasterizer keeps the largest z value. Our camera-space
        # depth is positive in front of the camera, so negate it to make nearer
        # surfaces win and match pyrender/OpenGL visibility.
        face_vertices_z = -depth[:, faces_t]
        valid_faces = (depth[:, faces_t] > 1e-4).all(dim=-1)

        face_features = self._kaolin_face_features(face_vertices_camera, mesh_base_color)
        image, face_idx = self._kaolin_mesh.rasterize(
            height,
            width,
            face_vertices_z,
            face_vertices_image,
            face_features,
            valid_faces=valid_faces,
            backend="cuda",
        )

        alpha = (face_idx >= 0).unsqueeze(-1).to(dtype)
        background = torch.tensor(scene_bg_color, device=device, dtype=dtype).view(1, 1, 1, 3)
        rgb = image * alpha + background * (1.0 - alpha)
        rgba = torch.cat([rgb, alpha], dim=-1)[0]
        return rgba.detach().cpu().numpy().astype(np.float32)

    def _kaolin_face_features(self, face_vertices_camera: torch.Tensor, mesh_base_color):
        v0, v1, v2 = face_vertices_camera.unbind(dim=2)
        normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        normals = torch.nn.functional.normalize(normals, dim=-1)

        key = torch.tensor([0.0, 0.0, 1.0], device=face_vertices_camera.device, dtype=face_vertices_camera.dtype)
        fill = torch.tensor([-0.35, -0.25, 0.75], device=face_vertices_camera.device, dtype=face_vertices_camera.dtype)
        fill = torch.nn.functional.normalize(fill, dim=0)
        key_shade = (normals * key.view(1, 1, 3)).sum(dim=-1).clamp(0.0, 1.0)
        fill_shade = (normals * fill.view(1, 1, 3)).sum(dim=-1).clamp(0.0, 1.0)
        shade = (0.35 + 0.45 * key_shade + 0.20 * fill_shade).clamp(0.0, 1.0)

        base_color = torch.tensor(mesh_base_color, device=face_vertices_camera.device, dtype=face_vertices_camera.dtype)
        face_color = (shade.unsqueeze(-1) * base_color.view(1, 1, 3)).clamp(0.0, 1.0)
        return face_color.unsqueeze(2).expand(-1, -1, 3, -1).contiguous()

    def render_rgba_multiple(
            self,
            vertices: List[np.array],
            cam_t: List[np.array],
            rot_axis=[1,0,0],
            rot_angle=0,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0,0,0),
            render_res=[256, 256],
            focal_length=None,
        ):

        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))

        mesh_list = [pyrender.Mesh.from_trimesh(self.vertices_to_trimesh(vvv, ttt.copy(), mesh_base_color, rot_axis, rot_angle)) for vvv,ttt in zip(vertices, cam_t)]

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        for i,mesh in enumerate(mesh_list):
            scene.add(mesh, f'mesh_{i}')

        camera_pose = np.eye(4)
        # camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        focal_length = focal_length if focal_length is not None else self.focal_length
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color

    def add_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

    def add_point_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses(dist=0.5)
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            # node = pyrender.Node(
            #     name=f"light-{i:02d}",
            #     light=pyrender.DirectionalLight(color=color, intensity=intensity),
            #     matrix=matrix,
            # )
            node = pyrender.Node(
                name=f"plight-{i:02d}",
                light=pyrender.PointLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)
