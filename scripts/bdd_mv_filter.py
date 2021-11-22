"""
    Extract the required types of images and corresponding labels from BDD dataset
    Generate sub-dataset similar to the BDD dataset catalog
    default extract types:
        daytime + clear + city_street
        night + clear + city_street
"""


import os
import json
import shutil

set_type = 'val'

json_dir_path = '/data1/zefeng/datasets/bdd100k/labels/100k/' + set_type    # bdd label path
img_dir_path = '/data1/zefeng/datasets/bdd100k/images/100k/' + set_type     # bdd image path
daytime_dir_path = '/data1/zefeng/datasets/bdd100k/coco/images_daytime/'                 # save subset root
night_dir_path = '/data1/zefeng/datasets/bdd100k/coco/images_night/'

# d_json_save_dir_path = daytime_dir_path + 'labels\\' + set_type     # save subset label path
# n_json_save_dir_path = night_dir_path + 'labels\\' + set_type
d_img_save_dir_path = daytime_dir_path + set_type      # save subset image path
n_img_save_dir_path = night_dir_path + set_type

# if os.path.exists(d_json_save_dir_path) and os.path.exists(n_json_save_dir_path):
#     pass
# elif os.path.exists(json_dir_path) and os.path.exists(img_dir_path):
#     os.mkdir(d_json_save_dir_path)
#     os.mkdir(n_json_save_dir_path)
#     os.mkdir(d_img_save_dir_path)
#     os.mkdir(n_img_save_dir_path)

daytime_clear, night_clear = 0, 0   # the number of timeofday with clear weather
"""
val:
    daytime_clear = 1764    daytime_clear_city = 933
    night_clear = 3274      night_clear_city = 2133
train:
    daytime_clear = 12477    daytime_clear_city = 6647
    night_clear = 22929      night_clear_city = 15090
"""

files = os.listdir(json_dir_path)   # read all json file
for file in files:
    json_file = os.path.join(json_dir_path,file)    # json path
    info = json.load(open(json_file))               # load json file
    if info['attributes']['timeofday'] == 'daytime' and info['attributes']['weather'] == 'clear' and info['attributes']['scene'] == 'city street':
        daytime_clear += 1
        # shutil.copy(json_file, d_json_save_dir_path)    # copy file.json to new dir
        img_name = file.split('.')[0] + '.jpg'          # images name
        img_file = os.path.join(img_dir_path, img_name) # image path
        shutil.copy(img_file, d_img_save_dir_path)      # copy file.jpg to new dir
        pass
    elif info['attributes']['timeofday'] == 'night' and info['attributes']['weather'] == 'clear' and info['attributes']['scene'] == 'city street':
        night_clear += 1
        # shutil.copy(json_file, n_json_save_dir_path)
        img_name = file.split('.')[0] + '.jpg'
        img_file = os.path.join(img_dir_path, img_name)
        shutil.copy(img_file, n_img_save_dir_path)
        pass
    else:
        continue

print(daytime_clear)    # number
print(night_clear)      # number
