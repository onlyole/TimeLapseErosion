
import sys, os
cloudcompy_cc_path = r'D:\Grothum\source\CloudComPy310_20230705\CloudComPy310\CloudCompare'
if cloudcompy_cc_path not in sys.path:
    sys.path.append(cloudcompy_cc_path)

sys.path.insert(1, r'/M3C2')
import cloudComPy as cc
from src.core.E4DPointcloud import E4DPointcloud
import numpy as np
import pandas as pd
import logging
import glob
import natsort
import re
import configparser
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from datetime import datetime
from sklearn.neighbors import NearestNeighbors

import cloudComPy as cc

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == '__main__':

    ### Configuration
    root_dir = r'D:\Grothum\Kamerstation_Metashape\Station_unten'
    cloud_import = os.path.join(root_dir, 'm3c2')
    outpath_geotiff = os.path.join(root_dir, 'm3c2', 'geotiff')
    transform_path = os.path.join(root_dir, 'm3c2', 'transform') # transformation, if available
    crop_path = os.path.join(root_dir, 'm3c2', 'crop') # directory with polygon files for cropping

    # create output directory for geotiffs
    if not os.path.exists(outpath_geotiff):
        os.mkdir(outpath_geotiff)

    # collect m3c2-clouds
    clouds = [x for x in glob.glob(os.path.join(cloud_import, '*.txt'))]

    # iterate through clouds
    for cl in tqdm(clouds):

        cloud = E4DPointcloud()
        cloud.load_pointcloud(cl, file_structure=E4DPointcloud.CloudFileStructure.XYZS)

        for file in glob.glob(os.path.join(crop_path, '*.txt')):
            cloud.crop_pointcloud(dim=2, points_filename=file, inside=True)

        # apply transformation if existent
        if os.path.exists(transform_path):
            for item in sorted(glob.glob(os.path.join(transform_path, '*.txt'))):
                transform = cloud.load_transformation(item).inverse()
                cloud.apply_transformation(transform)

        cloud.calculate_geotiff(gridstep=0.01, output_dir=outpath_geotiff)
