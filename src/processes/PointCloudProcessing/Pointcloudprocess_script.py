import os

from src.core.Pointcloudprocessing import PointCloudProcessing
from src.core.CloudPair import CloudPair
import logging
import glob
import natsort
import definitions
import cloudComPy as cc



if __name__ == '__main__':



    # Configuration
    station = 'Station_oben'
    months = ['2020-{:02d}'.format(x) for x in range(6, 9)]

    root_dir = os.path.join(definitions.DEMO_DIR, 'e4d_demo', 'Kamerstation_Metashape')

    skip_if_exist = True  # skip file, if it already exists, if False, recalculate.

    root_m3c2_config_file = os.path.join(root_dir, 'm3c2_params_2022-08-02.txt') # m3c2 parameters from cloud compare
    cutting_polyline_dir = os.path.join(root_dir, station, 'cropPolygons') # crop polygon directory (Crop2D)

    for mth in months:

        cloud_dir = os.path.join(root_dir, station, mth, 'Dense')
        pt_dir = os.path.join(root_dir, station, mth, 'ptPrecision')
        log_dir = os.path.join(root_dir, station, mth, 'logFiles')

        cloud_paths = natsort.natsorted([x for x in glob.glob(rf'{cloud_dir}\*.ply')])
        pt_paths = natsort.natsorted([x for x in glob.glob(rf'{pt_dir}\*.txt')])
        log_paths = natsort.natsorted([x for x in glob.glob(rf'{log_dir}\*.log')])

        clouds = []
        for dense in cloud_paths:
            p = CloudPair()
            p.dense = dense

            prec = [pr for pr in pt_paths if p.date in pr]
            log = [lg for lg in log_paths if p.date in lg]

            if prec and log:
                p.prec = prec[0]
                p.log = log[0]
                clouds.append(p)

        for idx, cl_pair in enumerate(clouds):
            try:
                outpath = os.path.join(os.path.dirname(cl_pair.dense), 'filtered',
                                       f"{os.path.splitext(os.path.basename(cl_pair.dense))[0]}.txt")
                if skip_if_exist and os.path.exists(outpath):
                    continue

                processing = PointCloudProcessing(cl_pair)
                processing.crop_files = [file for file in glob.glob(os.path.join(cutting_polyline_dir, '*.txt'))]

                # filter precision maps bases on neighborhood
                for field in ['sX(mm)', 'sY(mm)', 'sZ(mm)']:
                    processing.filter_pointcloud_on_scalarfield(N_nbnr=20,
                                                                r=0.5,
                                                                filter_variable=field,
                                                                multiplier_thresh_std=2)

                # crop dense cloud
                processing.crop_dense()
                # spatial subsampling
                processing.pc.spatial_subsampling()
                # plane fit and filtering by distance to plane
                params = cc.Cloud2MeshDistancesComputationParams()
                params.signedDistances = True
                processing.pc.filter_distance_to_plane(params=params, sigma_neg=3.0, sigma_pos=3.0)
                # noise filter
                processing.pc.noise_filter_params.knn = 30
                processing.pc.noise_filter_params.use_knn = True
                processing.pc.noise_filter_params.sigma = 2.0
                processing.pc.noise_filter_params.remove_isolated_points = True
                processing.pc.filter_noise()
                # substract slope or apply transformation of first cloud
                if not os.path.exists(os.path.join(root_dir, station, 'transform', 'transform.txt')):
                    processing.pc.remove_slope()
                    transform = processing.pc.transform_matrix
                    processing.pm_cloud.apply_transformation(transform)
                    processing.pm_cloud.transform_scalar_fields(['sX(mm)', 'sY(mm)', 'sZ(mm)'], transform)
                    processing.pc.save_transformation(filename=os.path.join(root_dir, station,
                                                                            'transform', 'transform.txt'),
                                                      transform=transform)
                else:
                    # transform slope-correction from first point cloud
                    transform = processing.pc.load_transformation(os.path.join(root_dir, station,
                                                                               'transform', 'transform.txt'))
                    processing.pc.apply_transformation(transform)
                    processing.pm_cloud.transform_scalar_fields(['sX(mm)', 'sY(mm)', 'sZ(mm)'], transform)
                    processing.pm_cloud.apply_transformation(transform)
                # raster dense
                processing.raster_dense(grid_step=0.005)
                # interpolate prec on dense
                processing.interpolate_prec_on_dense(radius=1.0, sigma=0.3)
                # processing.interpolate_prec_on_dense(radius=0.15, sigma=0.06) # for prag feld
                # remove unneccesary scalar fields
                processing.pc.remove_scalar_fields(
                    ['class', 'confidence', 'Cell height values', 'covXX(m2)', 'covXY(m2)', 'covXZ(m2)',
                     'covYY(m2)',
                     'covYZ(m2)', 'covZZ(m2)'])
                # export filtered dense cloud
                os.makedirs(os.path.dirname(outpath)) if not os.path.exists(os.path.dirname(outpath)) else None
                outframe = processing.pc.cloud_to_dataframe()
                outframe = outframe.drop('Alpha', axis=1, errors='ignore')
                outframe.to_csv(outpath, index=False, sep=',', float_format='%.3f', na_rep='NaN')
            except Exception as e:
                print(e)
                continue
