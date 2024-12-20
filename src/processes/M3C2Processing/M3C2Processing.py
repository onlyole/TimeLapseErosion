
import os
from src.core.CloudPair import CloudPair

from src.core.E4DPointcloud import E4DPointcloud
from src.core.Pointcloudprocessing import create_m3c2_config_file
import pandas as pd
import logging
import glob
import natsort
import re
import definitions
from tqdm import tqdm
from datetime import datetime



if __name__ == '__main__':

    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)

    # Configuration
    station = 'Station_oben'
    root_dir = os.path.join(definitions.DEMO_DIR, 'e4d_demo', 'Kamerstation_Metashape')
    outpath_m3c2 = os.path.join(definitions.DEMO_DIR, 'e4d_demo', 'Kamerstation_Metashape', station, 'm3c2')

    set_registration_error = False  # set registration error from point cloud processing files, if True
    root_m3c2_config_file = os.path.join(root_dir, 'm3c2_params_2022-08-02.txt')


    ### M3C2 Processing chain

    clouds = []
    # collect processing pointclouds and log files
    cloud_dir = os.path.join(root_dir, station, '????-??', 'Dense', 'filtered')
    log_dir = os.path.join(root_dir, station, '????-??', 'logFiles')
    m3c2_params_dir = os.path.join(root_dir, station, '????-??', 'm3c2Params')

    months_paths = [x for x in glob.glob(os.path.join(root_dir, '????-??'))]
    # create m32cParams, if not exist
    for mth in months_paths:
        if not os.path.exists(os.path.join(root_dir, mth, 'm3c2Params')):
            os.mkdir(os.path.join(root_dir, mth, 'm3c2Params'))

    # create m3c2 solution dir, if not exist
    if not os.path.exists(outpath_m3c2):
        os.mkdir(outpath_m3c2)

    # sort cloud paths and log paths numerically
    cloud_paths = natsort.natsorted([x for x in glob.glob(rf'{cloud_dir}\*.txt')])
    log_paths = natsort.natsorted([x for x in glob.glob(rf'{log_dir}\*.log')])


    for dense in cloud_paths:
        p = CloudPair()
        p.dense = dense
        p.log = [x for x in log_paths if p.date in x][0]
        # generate m3c2 processing params with reference error
        dense_month = re.search("\d{4}-\d{2}", dense).group()
        m3c2_actual_dir = m3c2_params_dir.replace('????-??', dense_month)
        m3c2_param_path = os.path.join(m3c2_actual_dir,
                                       f'm3c2_params_{os.path.splitext(os.path.basename(dense))[0]}.txt')
        if os.path.exists(m3c2_param_path):
            create_m3c2_config_file(root_m3c2_config_file, p.log, m3c2_param_path)
        p.m3c2_param = m3c2_param_path
        clouds.append(p)

    # sort clouds based on date
    clouds.sort(key=lambda x: datetime.strptime(x.date, '%Y-%m-%dT%H-%M-%S'))

    # set reference cloud
    ref_start_index = 0
    ref_cloud = E4DPointcloud()
    ref_cloud.load_pointcloud(clouds[ref_start_index].dense, file_structure=E4DPointcloud.CloudFileStructure.XYZS)
    ref_time = clouds[ref_start_index].date
    ref_index = clouds[ref_start_index].index
    # set m3c2 processing params
    ref_cloud.m3c2_processing_params_file = root_m3c2_config_file

    # import reset reference file
    reset_file = r"D:\Grothum\Kamerstation_Metashape\reset_reference.txt"
    reset_df = pd.read_csv(reset_file, header=0, delimiter=',')
    reset_df["date"] = pd.to_datetime(reset_df["date"], format="%Y-%m-%d")
    reset_current_idx = 0

    for cl in tqdm(clouds):
        try:

            if pd.to_datetime(cl.date, format="%Y-%m-%dT%H-%M-%S") > reset_df['date'].iloc[reset_current_idx]:
                ref_cloud = E4DPointcloud()
                ref_cloud.load_pointcloud(cl.dense, file_structure=E4DPointcloud.CloudFileStructure.XYZS)
                ref_time = cl.date
                ref_index = cl.index
                while pd.to_datetime(cl.date, format="%Y-%m-%dT%H-%M-%S") > reset_df['date'].iloc[reset_current_idx]:
                    reset_current_idx += 1

            sub_time = cl.date
            sub_index = cl.index
            outpath = os.path.join(
            os.path.join(outpath_m3c2, f'm3c2_{ref_index}_{ref_time}_to_{sub_index}_{sub_time}.txt'))

            if os.path.exists(outpath):
                continue

            sub_cloud = E4DPointcloud()
            sub_cloud.load_pointcloud(cl.dense, file_structure=E4DPointcloud.CloudFileStructure.XYZS)



            # set m3c2 params of actual cloud file
            if set_registration_error:
                ref_cloud.m3c2_processing_params_file = cl.m3c2_param
            else:
                ref_cloud.m3c2_processing_params_file = root_m3c2_config_file

            # find indexes of precission scalarfields
            ref_prec_idx = []
            sub_prec_idx = []
            ref_dict = ref_cloud.pointcloud.getScalarFieldDic()
            sub_dict = sub_cloud.pointcloud.getScalarFieldDic()

            for field in ['sX(mm)', 'sY(mm)', 'sZ(mm)']:
                ref_prec_idx.append(ref_dict.get(field, None))
                sub_prec_idx.append(sub_dict.get(field, None))

            # m3c2_result = ref_cloud.calculate_m3c2_pm(sub_cloud, ref_cloud, ref_prec_idx, sub_prec_idx,
            #                                           [0.001, 0.001, 0.001])
            m3c2_result = ref_cloud.calculate_m3c2_pm(sub_cloud, ref_cloud,
                                                      scalar_field_index_ref_cloud=['sX(mm)', 'sY(mm)', 'sZ(mm)'],
                                                      scalar_field_index_target_cloud=['sX(mm)', 'sY(mm)', 'sZ(mm)'],
                                                      scale=[0.001, 0.001])

            outframe = m3c2_result.cloud_to_dataframe() # if m3c2_result is obtainable

            if outframe is not None:
                outframe.to_csv(outpath, index=False, sep=',', na_rep='NaN', float_format='%.5f')

        except Exception as e:
            print(e)
            continue

