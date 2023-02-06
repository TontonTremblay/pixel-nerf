import json
import os
from os import path


import numpy as np
from PIL import Image
import glob 
import argparse
import subprocess
import cv2 
import torch
import math 


import meshcat
import meshcat.geometry as g
import meshcat.transformations as mtf

import pyrr 
import simplejson as json


def create_visualizer(clear=True, zmq_url='tcp://127.0.0.1:6000'):
    """
    If you set zmq_url=None it will start a server
    """

    print('Waiting for meshcat server... have you started a server? Run `meshcat-server` to start a server')
    vis = meshcat.Visualizer(zmq_url=zmq_url)
    if clear:
        vis.delete()
    return vis


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
            else:
                color = np.random.randint(low=0,high=256,size=3)
                data['color'] = color
        else:
            color = [0,255,0]

        mesh_vis = trimesh_to_meshcat_geometry(data['mesh_transformed'])
        color_hex = rgb2hex(tuple(color))
        material = meshcat.geometry.MeshPhongMaterial(color=color_hex)
        vis[name].set_object(mesh_vis, material)



def make_frame(vis, name, T=None, h=0.15, radius=0.001, o=1.0):
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

  if T is not None:
      print(T)
      vis[name].set_transform(T)


def visualize_pointcloud(vis, name, pc, color=None, transform=None, **kwargs):
    """
    Args:
        vis: meshcat visualizer object
        name: str
        pc: Nx3 or HxWx3
        color: (optional) same shape as pc [0-255] scale or just rgb tuple
        transform: (optional) 4x4 homogeneous transform
    """
    if pc.ndim == 3:
        pc = pc.reshape(-1, pc.shape[-1])

    if color is not None:
        if isinstance(color, list):
            color = np.array(color)
        color = np.array(color)
        if color.ndim == 3:
            color = color.reshape(-1, color.shape[-1])/255.0
        if color.ndim == 1:
            color = np.ones_like(pc) * np.array(color)/255.0
    else:
        color = np.ones_like(pc)

    vis[name].set_object(
        meshcat.geometry.PointCloud(position=pc.T, color=color.T, **kwargs))

    if transform is not None:
        vis[name].set_transform(transform)





# vis = create_visualizer()



# run with OPENCV_IO_ENABLE_OPENEXR=1 python ....


# OPENCV_IO_ENABLE_OPENEXR=1 python convert_nvisii.py --blenderdir /media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/google_shoe_single_3-9-22/ --outdir /media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/google_shoe_single_3-9-22_PIXELNERF/


parser = argparse.ArgumentParser(description="Compare group of runs.")

parser.add_argument("--blenderdir", 
  # default="/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/shapenet/renders/img/03001627/", 
  # default="/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/shapenet/renders/img/03001627/", 
  # default="/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/shapenet/renders_table/img/04379243/",
  default="/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/shapenet/renders_car/img/02958343/",
  help="guess folder"
)
parser.add_argument("--outdir", 
  # default="/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/shapenet/pixelner/", 
  # default="/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/shapenet/pixelnerf_table/", 
  default="/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/shapenet/pixelnerf_cars/", 
  help="gt folder")
parser.add_argument("--size", default=128, help="gt folder")
# parser.add_argument("--out", default="exp.csv", help="out_file")
# parser.add_argument("--set", default="test", help="out_file")

# splits
# /media/jtremblay/data_large/code/get3d/3dgan_data_split/shapenet_chair/

opt = parser.parse_args()



blenderdir = opt.blenderdir
outdir = opt.outdir
n_down = opt.size

print(outdir)
if not os.path.exists(outdir):
  os.makedirs(outdir)

# structure folders
# images_XXX, XXX 

# files
# metadata.json
# transform_XXX.json
folders = ['_train','_test','_val']
for folder in folders:
  if not os.path.exists(outdir + '/'+folder):
    os.makedirs(outdir + '/'+folder)
    subprocess.call(['mkdir',outdir + '/'+folder])

#generete the json files
out_train = {}
out_val = {}
out_test = {}
scale = 1

out = out_train
folder_name_out = '_train'
folders = sorted(glob.glob(blenderdir+"*/"))


# camera intrinsics

# fovy = np.arctan(32 / 2 / 35) * 2
cam_x_angle = 0.8575560450553894 / np.pi * 180.0

fov = math.degrees(math.atan((128/(2* 0.8575560450553894)))*2)
fov = 140
print(math.degrees(0.8575560450553894),cam_x_angle,math.degrees(fov))

# raise()

def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def create_my_world2cam_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)

    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

    new_t = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    new_t[:, :3, 3] = -origin
    new_r = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    new_r[:, :3, :3] = torch.cat(
        (left_vector.unsqueeze(dim=1), up_vector.unsqueeze(dim=1), forward_vector.unsqueeze(dim=1)), dim=1)
    world2cam = new_r @ new_t
    return world2cam

def create_camera_from_angle(phi, theta, sample_r=1.2, device='cpu'):
    '''
    :param phi: rotation angle of the camera
    :param theta:  rotation angle of the camera
    :param sample_r: distance from the camera to the origin
    :param device:
    :return:
    '''
    phi = torch.tensor([phi])
    theta = torch.tensor([theta])
    phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)

    camera_origin = torch.zeros((1,3))
    camera_origin[:, 0:1] = sample_r * torch.sin(phi) * torch.cos(theta)
    camera_origin[:, 2:3] = sample_r * torch.sin(phi) * torch.sin(theta)
    camera_origin[:, 1:2] = sample_r * torch.cos(phi)

    forward_vector = normalize_vecs(camera_origin)

    world2cam_matrix = create_my_world2cam_matrix(forward_vector, camera_origin, device=device)
    w2c = world2cam_matrix[0].numpy()
    # print(camera_origin)
    # w2c[0,-1] = camera_origin[0,0]
    # w2c[1,-1] = camera_origin[0,1]
    # w2c[2,-1] = camera_origin[0,2]

    # rotate 

    w2c = pyrr.Matrix44(w2c)
    w2c = pyrr.matrix44.inverse(w2c)
    w2c = w2c * pyrr.matrix44.create_from_axis_rotation([1,0,0],-np.pi/2)
    return w2c

# load the files from txt file

# also create the transform json

for i_folder, folder_name in enumerate(folders):
  name = folder_name.split("/")[-2]
  path_out = f"{outdir}/{folder_name_out}/{name}/"

  subprocess.call(['mkdir',path_out])
  subprocess.call(['mkdir',path_out+"/rgb"])
  subprocess.call(['mkdir',path_out+"/pose"])

  files_to_load = sorted(glob.glob(folder_name+"*.png"))

  rotation_camera = np.load(os.path.join(f'{folder_name.replace("img","camera")}/rotation.npy'))
  elevation_camera = np.load(os.path.join(f'{folder_name.replace("img","camera")}/elevation.npy'))

  # load the json
  with open(f"{folder_name}/transforms.json", 'r') as f:
    data_scene = json.load(f)

  for i_file, file_path in enumerate(files_to_load):
    name = file_path.split('/')[-1]

    s_output = f"{fov} {128/2} {128/2}\n0 0 0 \n{128} {128}"

    with open(path_out+'intrinsics.txt','w+')as f:
      f.write(s_output)

    c2w = data_scene['frames'][i_file]['transform_matrix']
    c2w = np.array(c2w).T
    # print(c2w)
    # print(c2w2)

    # make_frame(vis,str(i_file)+str(i_folder),c2w2,radius=0.01)


    s_output = f"{c2w[0][0]} {c2w[1][0]} {c2w[2][0]} {c2w[3][0]} {c2w[0][1]} {c2w[1][1]} {c2w[2][1]} {c2w[3][1]} {c2w[0][2]} {c2w[1][2]} {c2w[2][2]} {c2w[3][2]} {c2w[0][3]} {c2w[1][3]} {c2w[2][3]} {c2w[3][3]}"
    print(s_output)
    with open(path_out+"/pose/"+f'{name.replace(".png","")}.txt','w+')as f:
      f.write(s_output)
    print(s_output)

    # raise()
    subprocess.call(
      [
      "ln",'-s',file_path,path_out + "rgb/"+name
      ]
      )
    # raise()
    # im = load_rgb_exr(file_path.replace("json","exr"),resize=n_down)
    # # print(im.shape)
    # im_color = im[:,:,:3]
    # alpha = im[:,:,3]
    # white = np.ones(im_color.shape)*255
    # alpha_tmp = np.zeros(im_color.shape)
    # alpha_tmp[:,:,0] = alpha
    # alpha_tmp[:,:,1] = alpha
    # alpha_tmp[:,:,2] = alpha

    # im_color = (white * (1-alpha_tmp/255.0)) +im_color
    # cv2.imwrite(f"{path_out}/rgb/{str(i_file).zfill(6)}.png",im_color)
    # raise()
  # raise()

