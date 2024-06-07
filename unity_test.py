# import sys
# import os
# import subprocess
#
# def main():
#     # 接收传入的参数
#     input_string = sys.argv[1]
#     input_int = int(sys.argv[2])
#     input_float = float(sys.argv[3])
#
#     print(f"Received string: {input_string}")
#     print(f"Received int: {input_int}")
#     print(f"Received float: {input_float}")
#
#     # 输出当前的运行路径
#     current_path = os.getcwd()
#     print(f"Current working directory: {current_path}")
#
#     # 输出当前使用的 Python 解释器路径
#     print(f"Python executable: {sys.executable}")
#
#     # 输出环境变量（可以注释掉这一部分，如果输出太多）
#     # print("Environment variables:")
#     # for key, value in os.environ.items():
#     #     print(f"{key}: {value}")
#
#     # 输出已安装的 pip 包
#
#     # 进行一些处理，例如将整数加倍，将浮点数平方
#     result_int = input_int * 2
#     result_float = input_float ** 2
#
#     # 输出结果
#     print(f"Result int: {result_int}")
#     print(f"Result float: {result_float}")
#
# if __name__ == "__main__":
#     main()
import tqdm

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
from hloc.pipelines.Cambridge.utils import create_query_list_with_fixed_intrinsics, evaluate,create_query_list_with_fixed_intrinsics_use_imagedir
from hloc.utils import viz_3d
from pprint import pformat
import pdb
import time


# images = Path("datasets/sacre_coeur_test")
colmap_images = Path("datasets/jet18/all_images/colmap_images")
query_images = Path("datasets/jet18/all_images/query")
# all_images = Path("datasets/jet18/all_images")
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
feature_conf['model']['max_keypoints']  = 5000
matcher_conf = match_features.confs["disk+lightglue"]
# list the standard configurations available
retrieval_conf = extract_features.confs["netvlad"]
print(f"Configs for feature extractors:\n{pformat(extract_features.confs)}")
print(f"Configs for feature matchers:\n{pformat(match_features.confs)}")  # list the standard configurations available

if __name__ == "__main__":
    # pdb.set_trace()
    # 提取local features,  对重建图片,  生成 "features.h5"
    # features = extract_features.old_main(feature_conf, colmap_images, feature_path=features)
    print('zqk')
    # pdb.set_trace()