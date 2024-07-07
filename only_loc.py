import tqdm
import torch

a = torch.cuda.is_available()
print(torch.__version__)
print(a)

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
colmap_images = Path("datasets/jet18/all_images/colmap_images")
query_images = Path("datasets/jet18/all_images/query")
my_model = Path("datasets/jet18/my_model")
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
print(f"Configs for feature extractors:\n{pformat(extract_features.confs)}")
print(f"Configs for feature matchers:\n{pformat(match_features.confs)}")  # list the standard configurations available

reference_sfm = outputs / "sfm_superpoint+superglue"  # the SfM model we will build  这个基本作为定位的参考地图

################################################################################################################################

# #
# # pdb.set_trace()
#
# reference_sfm = outputs / "sfm_superpoint+superglue"  # the SfM model we will build  这个基本作为定位的参考地图
#
# # 针对 78 张图片，借鉴my_model的相机位置，进行三角化       生成 sfm_superpoint+superglue
# reconstruction = triangulation.main(
#     reference_sfm, my_model, colmap_images, sfm_pairs, features, matches
# )
# pdb.set_trace()
#
# # pdb.set_trace()
# fig = viz_3d.init_figure()
# viz_3d.plot_reconstruction(
#     fig, reconstruction, color="rgba(255,0,0,0.5)", name="mapping", points_rgb=True
# )
# fig.show()
################################################################################################################################
# 加载模型
feature_model = extract_features.extract_init(feature_conf)
retrieval_model = extract_features.extract_init(retrieval_conf)

all_query_images = get_image_filenames(query_images)
for one_query_image in all_query_images:
    start_time = time.time()
    print('\n\n\n\n\n', one_query_image)

    start_time1 = time.time()
    # 针对需要定位的3帧，提取局部特征，总共提取3张照片，生成生成  "features.h5"  106M
    features = extract_features.main(feature_conf, query_images, feature_path=features, overwrite=True,
                                     image_list=[one_query_image],
                                     model=feature_model)
    end_time1 = time.time()
    execution_time = (end_time1 - start_time1) * 1000  # 将秒转换为毫秒
    print("Function execution time: nd_time1 - start_time1   ", execution_time, "milliseconds")

    # 针对需要定位的3帧，提取全局描述子，并添加到，，，生成"globals_features.h5"   831K==============主要耗时4.5s,总共才6.4s
    global_descriptors = extract_features.main(retrieval_conf, query_images, feature_path=globals_features,
                                               image_list=[one_query_image],
                                               overwrite=True, model=retrieval_model)
    end_time2 = time.time()
    execution_time = (end_time2 - end_time1) * 1000  # 将秒转换为毫秒
    print("Function execution time: end_time2 - end_time1  ", execution_time, "milliseconds")

    # 然后对需要查询的图片，进行提取共视信息,,,,,生成pairs-loc.txt
    pairs_from_retrieval.main(
        global_descriptors, loc_pairs, num_matched=3, db_prefix="IMG", query_prefix="query"
    )
    end_time3 = time.time()
    execution_time = (end_time3 - end_time2) * 1000  # 将秒转换为毫秒
    print("Function execution time: end_time3 - end_time2 ", execution_time, "milliseconds")

    #  针对 3张查询的图片，每张有10个共视   "loc_matches.h5"
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, features, matches=loc_matches
    )
    end_time4 = time.time()
    execution_time = (end_time4 - end_time3) * 1000  # 将秒转换为毫秒
    print("Function execution time: end_time4 - end_time3 ", execution_time, "milliseconds")

    query_list = outputs / 'query_list_with_intrinsics.txt'
    import pycolmap

    camera = pycolmap.Camera(
        model='SIMPLE_RADIAL',
        width=1920,
        height=1080,
        params=[1919.0, 972.0, 540.0, 0.02103697391591438],
    )
    # 生成  query_list_with_intrinsics.txt
    # 这一步是在colmap模型中查询 要重定位图片的相机内参，这不就是要求，colmap模型中必须要有需要查询的图片吗？
    # 我重写了这个函数，这里我全部用第一张图片的内参，代替，
    # create_query_list_with_fixed_intrinsics(my_model, query_list, test_list)
    create_query_list_with_fixed_intrinsics_one_image(my_model, query_list, list_file=[one_query_image])
    # create_query_list_with_another_intrinsics_one_image(camera,query_list,list_file=[one_query_image])
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
    end_time5 = time.time()
    execution_time = (end_time5 - end_time4) * 1000  # 将秒转换为毫秒
    print("Function execution time: end_time5 - end_time4 ", execution_time, "milliseconds")

    end_time = time.time()

    execution_time = (end_time - start_time) * 1000  # 将秒转换为毫秒
    print("Function execution time:", execution_time, "milliseconds")

    pdb.set_trace()
    # show relocation
###################################################################################################################
# import pycolmap
#
# loc_key = loc_result.keys()
# for loc_key, loc_xyz in loc_result.items():
#     pose = pycolmap.Image(cam_from_world=loc_xyz)
#     # 获取第一个键值对
#     #     first_key, first_value = next(iter(reconstruction.cameras.items()))
#     # camera = reconstruction.cameras[first_key]
#     # camera = pycolmap.Camera(
#     #     model='SIMPLE_RADIAL',
#     #     width=3648,
#     #     height=2736,
#     #     params=[2851.03, 1824.0, 1338.0, 0.02603697391591438],
#     # )
#
#     # camera = pycolmap.Camera(
#     #     model='SIMPLE_RADIAL',
#     #     width=972,
#     #     height=540,
#     #     params=[1919.0, 972.0, 540.0, 0.02103697391591438],
#     # )
#     # 3648 2736 2851.037070234661 1824.0 1368.0 0.026471889620124405
#     viz_3d.plot_camera_colmap(
#         fig, pose, camera, color="rgba(0,255,0,0.5)", name=loc_key, fill=True
#     )
#
# fig.show()
###################################################################################################################


print(loc_result)
# all_images_in = Path("datasets/sacre_coeur_test/all_images_in")
# visualization.visualize_loc(
#     results, all_images_in, reference_sfm, n=1, top_k_db=1, prefix="query_", seed=2
# )
#
#
# import pycolmap
# loc_key = loc_result.keys()
# for loc_key,loc_xyz in loc_result.items():
#
#     pose = pycolmap.Image(cam_from_world=loc_xyz)
# # 获取第一个键值对
# #     first_key, first_value = next(iter(reconstruction.cameras.items()))
#     # camera = reconstruction.cameras[first_key]
#     # camera = pycolmap.Camera(
#     #     model='SIMPLE_RADIAL',
#     #     width=3648,
#     #     height=2736,
#     #     params=[2851.03, 1824.0, 1338.0, 0.02603697391591438],
#     # )
#
#     camera = pycolmap.Camera(
#         model='SIMPLE_RADIAL',
#         width=972,
#         height=540,
#         params=[1919.0, 972.0, 540.0, 0.02103697391591438],
#     )
# # 3648 2736 2851.037070234661 1824.0 1368.0 0.026471889620124405
#     viz_3d.plot_camera_colmap(
#         fig, pose, camera, color="rgba(0,255,0,0.5)", name=loc_key, fill=True
#     )
#
# fig.show()
# pdb.set_trace()
