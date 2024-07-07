from pathlib import Path
import numpy as np

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
    pairs_from_covisibility,
    pairs_from_retrieval,
    triangulation,
    localize_sfm
)
from hloc.visualization import plot_images, read_image
from hloc.pipelines.Cambridge.utils import create_query_list_with_fixed_intrinsics, evaluate, \
    create_query_list_with_fixed_intrinsics_use_imagedir, create_query_list_with_fixed_intrinsics_one_image, \
    create_query_list_with_another_intrinsics_one_image

from hloc.utils import viz_3d
from pprint import pformat
import pdb
import time
import os


def get_image_filenames(directory):
    image_filenames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(
                ".gif"):
            image_filenames.append(filename)
    return image_filenames


# images = Path("datasets/sacre_coeur_test")
colmap_images = Path("datasets/jet720p/images")
query_images = Path("/home/zqk/桌面/unity_photo")
my_model = Path("datasets/jet720p/my_model")
outputs = Path("outputs/demo/")

sfm_pairs = outputs / "pairs-sfm.txt"
loc_pairs = outputs / "pairs-loc.txt"
sfm_dir = outputs / "sfm"
features = outputs / "features.h5"
matches = outputs / "matches.h5"
loc_matches = outputs / "loc_matches.h5"
globals_features = outputs / "globals_features.h5"

feature_conf = extract_features.confs["disk"]
feature_conf['model']['max_keypoints'] = 5000
matcher_conf = match_features.confs["disk+lightglue"]
# list the standard configurations available
retrieval_conf = extract_features.confs["netvlad"]

reference_sfm = outputs / "sfm_superpoint+superglue"  # the SfM model we will build  这个基本作为定位的参考地图

# 针对需要定位的3帧，提取局部特征，总共提取3张照片，生成生成  "features.h5"  106M
features = extract_features.old_main(feature_conf, query_images, feature_path=features, overwrite=True)
# pdb.set_trace()
start_time = time.time()
# 针对需要定位的3帧，提取全局描述子，并添加到，，，生成"globals_features.h5"   831K==============主要耗时4.5s,总共才6.4s
global_descriptors = extract_features.old_main(retrieval_conf, query_images, feature_path=globals_features,
                                               overwrite=True)
# pdb.set_trace()
#
end_time = time.time()

execution_time = (end_time - start_time) * 1000  # 将秒转换为毫秒
print("Function execution time:", execution_time, "milliseconds")
# 然后对需要查询的图片，进行提取共视信息,,,,,生成pairs-loc.txt
pairs_from_retrieval.main(
    global_descriptors, loc_pairs, num_matched=3, db_prefix="2024", query_prefix="query"
)
# pdb.set_trace()

#  针对 3张查询的图片，每张有10个共视   "loc_matches.h5"
loc_matches = match_features.main(
    matcher_conf, loc_pairs, features, matches=loc_matches
)
# pdb.set_trace()

query_list = outputs / 'query_list_with_intrinsics.txt'
import pycolmap

camera = pycolmap.Camera(
    model='SIMPLE_RADIAL',
    width=1280,
    height=720,
    params=[695.47427324616319, 640.0, 360.0, -0.069179723657704695],
)
# camera = pycolmap.Camera(
#     model='SIMPLE_RADIAL',
#     width=3648,
#     height=2736,
#     params=[2851.0370702346609, 1824.0, 1368.0, 0.026471889620124405],
# )

# query_list_my = outputs / 'query_list_my.txt'
# test_list = my_model / 'list_test.txt'
# 生成  query_list_with_intrinsics.txt
# 这一步是在colmap模型中查询 要重定位图片的相机内参，这不就是要求，colmap模型中必须要有需要查询的图片吗？
# 我重写了这个函数，这里我全部用第一张图片的内参，代替，
# create_query_list_with_fixed_intrinsics(my_model, query_list, test_list)
# create_query_list_with_fixed_intrinsics_use_imagedir(my_model, query_list, image_dir=query_images)
create_query_list_with_another_intrinsics_one_image(camera, query_list, image_dir=query_images)
results = outputs / "loc.txt"  # the result file

loc_result = localize_sfm.main(
    reference_sfm,
    query_list,
    loc_pairs,
    features,
    loc_matches,
    results,
    covisibility_clustering=False,
)  # not required with SuperPoint+SuperGlue

print(loc_result)

