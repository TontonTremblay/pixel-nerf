import json
import os
from os import path


import numpy as np
from PIL import Image
import glob 
import argparse
import subprocess
import cv2 


# run with OPENCV_IO_ENABLE_OPENEXR=1 python ....


# OPENCV_IO_ENABLE_OPENEXR=1 python convert_nvisii.py --blenderdir /media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/google_shoe_single_3-9-22/ --outdir /media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/google_shoe_single_3-9-22_PIXELNERF/


parser = argparse.ArgumentParser(description="Compare group of runs.")

parser.add_argument("--blenderdir", default="", help="guess folder")
parser.add_argument("--outdir", default="", help="gt folder")
parser.add_argument("--size", default=128, help="gt folder")
parser.add_argument("--out", default="exp.csv", help="out_file")
parser.add_argument("--set", default="test", help="out_file")

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

#generete the json files
out_train = {}
out_val = {}
out_test = {}
scale = 1

out = out_train
folder_name_out = '_train'
folders = sorted(glob.glob(blenderdir+"*/"))

folders_new = []

folders_to_remove = [
  "Womens_Suede_Bahama_in_Graphite_Suede_t22AJSRjBOX",
  "Santa_Cruz_Mens_umxTczr1Ygg",
  "Great_Jones_Wingtip_kAqSg6EgG0I",
  "PureCadence_2_Color_HiRskRedNghtlfeSlvrBlckWht_Size_70",
  "MARTIN_WEDGE_LACE_BOOT",
  "Reebok_DMX_MAX_PLUS_RAINWALKER",
  "Reef_Star_Cushion_Flipflops_Size_8_Black",
  "JS_WINGS_20_BLACK_FLAG",
  "TZX_Runner",
  "Reebok_SH_PRIME_COURT_LOW",
  "Reebok_SH_PRIME_COURT_MID",
  "Timberland_Mens_Earthkeepers_Casco_Bay_Canvas_SlipOn",
  "Chelsea_BlkHeelPMP_DwxLtZNxLZZ",
  "ZX700_mzGbdP3u6JB",
  "Tiek_Blue_Patent_Tieks_Italian_Leather_Ballet_Flats",
  "F5_TRX_FG",
]
train_shoe_view = {
"Womens_Suede_Bahama_in_Graphite_Suede_t22AJSRjBOX":
94,
"Santa_Cruz_Mens_umxTczr1Ygg":
113,
"Great_Jones_Wingtip_kAqSg6EgG0I":
134,
"PureCadence_2_Color_HiRskRedNghtlfeSlvrBlckWht_Size_70":
126,
"MARTIN_WEDGE_LACE_BOOT":
104,
"Reebok_DMX_MAX_PLUS_RAINWALKER":
123,
"Reef_Star_Cushion_Flipflops_Size_8_Black":
117,
"JS_WINGS_20_BLACK_FLAG":
132,
"TZX_Runner":
106,
"Reebok_SH_PRIME_COURT_LOW":
101,
"Reebok_SH_PRIME_COURT_MID":
110,
"Timberland_Mens_Earthkeepers_Casco_Bay_Canvas_SlipOn":
137,
"Chelsea_BlkHeelPMP_DwxLtZNxLZZ":
80,
"ZX700_mzGbdP3u6JB":
129,
"Tiek_Blue_Patent_Tieks_Italian_Leather_Ballet_Flats":
106,
"F5_TRX_FG":
82,
}
for folder in folders:
  name = folder.split("/")[-2]
  keep = False

  for name_remove in folders_to_remove:
    if name_remove == name:
      keep = True
      print(name_remove,name)
  if keep:
    folders_new.append(folder)

print(len(folders_new),len(folders),len(folders)-len(folders_new))
# raise()

folder_name_out = '_test'

for i_folder, folder_name in enumerate(folders_new):
  # if i_folder>=280 and i_folder<282:
  #   folder_name_out = '_val'

  # elif i_folder>=282:
  #   out = out_test
  #   folder_name_out = '_test'
  path_out = f"{outdir}/{folder_name_out}/{str(i_folder).zfill(5)}/"
  # create folder
  subprocess.call(['mkdir',path_out])
  subprocess.call(['mkdir',path_out+"/rgb"])
  subprocess.call(['mkdir',path_out+"/pose"])

  # add intrinsics
  # scale = 1600/n_down
  scale = 800/n_down
  # fx = 1931.371337890625
  fx = 965.6856689453125
  s_output = f"{fx/scale} {n_down/2} {n_down/2}\n0 0 0 \n{n_down} {n_down}"

  with open(path_out+'intrinsics.txt','w+')as f:
    f.write(s_output)
  
  # files_to_load = sorted(glob.glob(folder_name+"*.json"))[:60]
  # files_to_load = sorted(glob.glob(folder_name+"*.json"))
  files_to_load = sorted(glob.glob(folder_name+"*.json"))[5:80]
  files_to_load.append(sorted(glob.glob(folder_name+"*.json"))[train_shoe_view[folder_name.split("/")[-2]]])
  files_to_load.append(sorted(glob.glob(folder_name+"*.json"))[train_shoe_view[folder_name.split("/")[-2]]])
  # print(folder_name.split("/")[-2],train_shoe_view[folder_name.split("/")[-2]],files_to_load[-1])
  print(folder_name.split("/")[-2],i_folder)
  continue
  for i_file, file_path in enumerate(files_to_load):
    if i_file < len(files_to_load)-2:
      continue
    with open(file_path) as json_file:
        data = json.load(json_file)
        c2w = data["camera_data"]["cam2world"]        

    print(c2w)
    # c2w = np.array(c2w).T
    s_output = f"{c2w[0][0]} {c2w[1][0]} {c2w[2][0]} {c2w[3][0]} {c2w[0][1]} {c2w[1][1]} {c2w[2][1]} {c2w[3][1]} {c2w[0][2]} {c2w[1][2]} {c2w[2][2]} {c2w[3][2]} {c2w[0][3]} {c2w[1][3]} {c2w[2][3]} {c2w[3][3]}"
    with open(path_out+"/pose/"+f'{str(i_file).zfill(6)}.txt','w+')as f:
      f.write(s_output)

    # make the png files
    # raise()
    def linear_to_srgb(img):
      limit = 0.0031308
      img = np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)
      img[img > 1] = 1
      img[img < 0] = 0
      return img

    def load_rgb_exr(img_path, resize=-1):
      img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
      # img[:, :, :3] = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
      if resize > 0:
        img = cv2.resize(img, (resize, resize))
      img = linear_to_srgb(img)
      img = np.int32(img * 255)
      
      return img


    im = load_rgb_exr(file_path.replace("json","exr"),resize=n_down)
    # print(im.shape)
    im_color = im[:,:,:3]
    alpha = im[:,:,3]
    white = np.ones(im_color.shape)*255
    alpha_tmp = np.zeros(im_color.shape)
    alpha_tmp[:,:,0] = alpha
    alpha_tmp[:,:,1] = alpha
    alpha_tmp[:,:,2] = alpha

    im_color = (white * (1-alpha_tmp/255.0)) +im_color
    cv2.imwrite(f"{path_out}/rgb/{str(i_file).zfill(6)}.png",im_color)
    # raise()

