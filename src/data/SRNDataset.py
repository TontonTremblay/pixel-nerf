import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
from util import get_image_to_tensor_balanced, get_mask_to_tensor

import numpy as np
import trimesh.transformations as tra


import meshcat
import meshcat.geometry as g
import meshcat.transformations as mtf


"""
Some code borrowed from https://github.com/google-research/ravens
under Apache license
"""


def isRotationMatrix(M, tol=1e-4):
    tag = False
    I = np.identity(M.shape[0])

    if (np.linalg.norm((np.matmul(M, M.T) - I)) < tol) and (np.abs(np.linalg.det(M) - 1) < tol):
        tag = True

    if (tag is False):
        print("M @ M.T:\n", np.matmul(M, M.T))
        print("det:", np.linalg.det(M))

    return tag


def trimesh_to_meshcat_geometry(mesh):
    """
    Args:
        mesh: trimesh.TriMesh object
    """

    return meshcat.geometry.TriangularMeshGeometry(mesh.vertices, mesh.faces)


def rgb2hex(rgb):
    """
    Converts rgb color to hex

    Args:
        rgb: color in rgb, e.g. (255,0,0)
    """
    return '0x%02x%02x%02x' % (rgb)


def visualize_scene(vis, object_dict, randomize_color=True):

    for name, data in object_dict.items():

        # try assigning a random color
        if randomize_color:
            if 'color' in data:
                color = data['color']

                # if it's not an integer, convert it to [0,255]
                if not np.issubdtype(color.dtype, np.int):
                    color = (color * 255).astype(np.int32)
            else:
                color = np.random.randint(low=0, high=256, size=3)
                data['color'] = color
        else:
            color = [0, 255, 0]

        # mesh_vis = trimesh_to_meshcat_geometry(data['mesh_transformed'])
        mesh_vis = trimesh_to_meshcat_geometry(data['mesh'])
        color_hex = rgb2hex(tuple(color))
        material = meshcat.geometry.MeshPhongMaterial(color=color_hex)

        mesh_name = f"{name}/mesh"
        vis[mesh_name].set_object(mesh_vis, material)
        vis[mesh_name].set_transform(data['T_world_object'])

        frame_name = f"{name}/transform"
        make_frame(vis, frame_name, T=data['T_world_object'])


def create_visualizer(clear=True):
    print('Waiting for meshcat server... have you started a server? Run `meshcat-server` to start a server')
    vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    if clear:
        vis.delete()
    return vis


def make_frame(vis, name, h=0.15, radius=0.001, o=1.0, T=None, transform=None):
    """Add a red-green-blue triad to the Meschat visualizer.
    Args:
      vis (MeshCat Visualizer): the visualizer
      name (string): name for this frame (should be unique)
      h (float): height of frame visualization
      radius (float): radius of frame visualization
      o (float): opacity
    """
    vis[name]['x'].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0xff0000, reflectivity=0.8, opacity=o))
    rotate_x = mtf.rotation_matrix(np.pi / 2.0, [0, 0, 1])
    rotate_x[0, 3] = h / 2
    vis[name]['x'].set_transform(rotate_x)

    vis[name]['y'].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0x00ff00, reflectivity=0.8, opacity=o))
    rotate_y = mtf.rotation_matrix(np.pi / 2.0, [0, 1, 0])
    rotate_y[1, 3] = h / 2
    vis[name]['y'].set_transform(rotate_y)

    vis[name]['z'].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0x0000ff, reflectivity=0.8, opacity=o))
    rotate_z = mtf.rotation_matrix(np.pi / 2.0, [1, 0, 0])
    rotate_z[2, 3] = h / 2
    vis[name]['z'].set_transform(rotate_z)

    if transform is not None:
        T = transform

    if T is not None:
        is_valid = isRotationMatrix(T[:3, :3])
        print(T)
        if not is_valid:
            raise ValueError(
                "meshcat_utils:attempted to visualize invalid transform T")

        vis[name].set_transform(T)






class SRNDataset(torch.utils.data.Dataset):
    """
    Dataset from SRN (V. Sitzmann et al. 2020)
    """

    def __init__(
        self, path, stage="train", image_size=(128, 128), world_scale=1.0,
    ):
        """
        :param stage train | val | test
        :param image_size result image size (resizes if different)
        :param world_scale amount to scale entire world by
        """
        super().__init__()
        self.base_path = path + "_" + stage
        self.dataset_name = os.path.basename(path)

        print("Loading SRN dataset", self.base_path, "name:", self.dataset_name)
        self.stage = stage
        assert os.path.exists(self.base_path)

        is_chair = "chair" in self.dataset_name
        if is_chair and stage == "train":
            # Ugly thing from SRN's public dataset
            tmp = os.path.join(self.base_path, "chairs_2.0_train")
            if os.path.exists(tmp):
                self.base_path = tmp

        self.intrins = sorted(
            glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
        )
        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()

        self.image_size = image_size
        self.world_scale = world_scale
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

        if is_chair:
            self.z_near = 1.25
            self.z_far = 2.75
        else:
            self.z_near = 0
            self.z_far = 3.1 
        self.lindisp = False

    def __len__(self):
        return len(self.intrins)

    def __getitem__(self, index):
        intrin_path = self.intrins[index]
        dir_path = os.path.dirname(intrin_path)
        rgb_paths_tmp = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
        pose_paths_tmp = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))

        rgb_paths = []
        pose_paths = []
        for i_rgb_path in range(len(rgb_paths_tmp)):
            rgb_path = rgb_paths_tmp[i_rgb_path]
            img = imageio.imread(rgb_path)[..., :3]
            mask = (img != 255).all(axis=-1)[..., None].astype(np.uint8) * 255
            rows = np.any(mask, axis=1)
            rnz = np.where(rows)[0]
            if len(rnz) == 0:
                continue
            rgb_paths.append(rgb_path)
            pose_paths.append(pose_paths_tmp[i_rgb_path])
        missing = 150 - len(rgb_paths)

        for m in range(missing):
            i_rand = np.random.randint(0,150-1-missing)
            rgb_paths.append(rgb_paths[i_rand])
            pose_paths.append(pose_paths[i_rand])

        assert len(rgb_paths) == len(pose_paths)

        with open(intrin_path, "r") as intrinfile:
            lines = intrinfile.readlines()
            try:
                focal, cx, cy = map(float, lines[0].split())
            except:
                focal, cx, cy, _ = map(float, lines[0].split())

            height, width = map(int, lines[-1].split())

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []

        # vis = create_visualizer(clear=True)
        i = 0 
        for rgb_path, pose_path in zip(rgb_paths, pose_paths):
            i+=1
            img = imageio.imread(rgb_path)[..., :3]
            img_tensor = self.image_to_tensor(img)
            mask = (img != 255).all(axis=-1)[..., None].astype(np.uint8) * 255
            mask_tensor = self.mask_to_tensor(mask)

            pose = torch.from_numpy(
                np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
            )
            # t = np.loadtxt(pose_path, dtype=np.float64)
            # poset = [[t[0],t[1],t[2],t[3]],
            #         [t[4],t[5],t[6],t[7]],
            #         [t[8],t[9],t[10],t[11]],
            #         [t[12],t[13],t[14],t[15]]]
            # make_frame(
            #     vis, str(i), transform=np.array(poset))
            # raise()
            # pose = pose @ self._coord_trans
            # make_frame(
            #     vis, str(i), transform=np.array(pose.double()))

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            # if len(rnz) == 0:
            #     raise RuntimeError(
            #         "ERROR: Bad image at", rgb_path, "please investigate!"
            #     )
            rmin, rmax = rnz[[0, -1]]
            cmin, cmax = cnz[[0, -1]]
            bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)

            all_imgs.append(img_tensor)
            all_masks.append(mask_tensor)
            all_poses.append(pose)
            all_bboxes.append(bbox)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_masks = torch.stack(all_masks)
        all_bboxes = torch.stack(all_bboxes)

        if all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            cx *= scale
            cy *= scale
            all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        if self.world_scale != 1.0:
            focal *= self.world_scale
            all_poses[:, :3, 3] *= self.world_scale
        focal = torch.tensor(focal, dtype=torch.float32)

        result = {
            "path": dir_path,
            "img_id": index,
            "focal": focal,
            "c": torch.tensor([cx, cy], dtype=torch.float32),
            "images": all_imgs,
            "masks": all_masks,
            "bbox": all_bboxes,
            "poses": all_poses,
        }
        return result
