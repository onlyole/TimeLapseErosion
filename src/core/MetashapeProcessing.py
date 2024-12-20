from __future__ import annotations

import os
import traceback
import sys
import csv
import time
import copy
import pandas as pd
import numpy as np
import src.core.Marker as Marker
import math
import matplotlib.pyplot as plt
import itertools
import re

from src.core.IOMeasurements import IOMeasurements

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# os.environ['agisoft_LICENSE'] = r"C:\Program Files\Agisoft\Metashape Pro\client.lic"
# os.environ['agisoft_LICENSE'] = r"C:\Program Files\Agisoft\Metashape Pro\client.lic"

import Metashape
import logging
import enum

logger = logging.getLogger()

formatter = logging.Formatter('[%(asctime)s][%(levelname)s: %(message)s]')
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)
logger.setLevel(logging.INFO)
logging_redirect_tqdm(logger)


# print(os.environ)

class MetashapeProcessing:

    def __init__(self):
        self.check_license()
        self.project_path: list[str] = None
        self.doc: Metashape.app.document = None
        self.image_list: pd.DataFrame = None
        self.calibration_list: pd.DataFrame = None
        self.cameras_list = []
        self.marker = Marker.Marker()

        self.align_params = MetashapeAlignmentParameters()

    def __del__(self):
        self.project_path: str = None
        self.doc: Metashape.app.document = None
        self.df: pd.DataFrame = None
        self.calibration_list: pd.DataFrame = None
        self.cameras_list = None

    class InputImageMode(enum.Enum):
        All = 1
        Daily = 2
        Events = 3

    def check_license(self):
        # check licence
        os.environ

        lic = Metashape.License()
        if lic.valid:
            print("Found valid license")
        else:
            print("No license found")
            sys.exit(-1)

    def setup_project(self):

        if self.project_path is None:
            logger.error(
                'No project-path has been set. Please do so by setting attribut: self.project_path = path/to/file.psx')
            raise IOError('Path "{path}" not readable')

        elif not os.path.isfile(self.project_path):
            logger.info(f'Project does not exist. Create new project on {self.project_path}')
            self.doc = Metashape.Document()
            self.doc.save(self.project_path)

        elif os.path.isfile(self.project_path):
            logger.info(f'Project exists. Open {self.project_path}')
            self.doc = Metashape.Document()
            self.doc.open(self.project_path)

    def reset_project(self):
        for chunk in self.doc.chunks: self.doc.remove(chunk)

    def save_project(self):

        if os.path.isfile(self.project_path):
            self.doc.save(self.project_path)
            logger.info(f'Saved to {self.project_path}')

    def read_input_image_attribution(self, filename: str):
        df = pd.read_csv(filename, sep=',', header=0, index_col=[0])
        df.fillna('', inplace=True)
        logger.info(f'Read image attribution file: {filename} with shape: {df.shape}')

        return df

    def read_input_image_attribution_daily(self, filename: str) -> pd.DataFrame:

        # import csv time attriubtuon
        self.df = pd.read_csv(filename, sep=',', header=0, index_col=[0])
        self.df.fillna('', inplace=True)

        # select second column and transform them into readable time
        ref_epoch = pd.Timestamp(1970, 1, 1, 0)

        cols = [col for col in self.df.columns]
        date_col = cols[1]
        self.df[date_col] = pd.to_datetime(self.df[cols[1]], unit='s')

        # Extract the date and time components
        self.df['date'] = self.df[date_col].dt.date
        self.df['time'] = self.df[date_col].dt.time

        # Filter rows that are closest to 10 a.m. for each day
        filtered_rows = []
        for date, group in self.df.groupby('date'):
            closest_time = min(group['time'], key=lambda x: abs(x.hour - 10))
            closest_row = group[group['time'] == closest_time].iloc[0]
            filtered_rows.append(closest_row)

        # Create a DataFrame from the filtered rows
        result_df = pd.DataFrame(filtered_rows)
        result_df[date_col] = (result_df[date_col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        result_df = result_df.drop(['date', 'time'], axis=1)
        # Set the aggregated DataFrame to member
        return result_df

    def cluster_images(self, time_attribution: pd.DataFrame, timedelta_threshold: pd.Timedelta) -> pd.DataFrame:

        cols = [col for col in time_attribution.columns]
        date_col_idx = 1
        date_col = cols[date_col_idx]
        time_attribution[date_col] = pd.to_datetime(time_attribution[date_col], unit='s')

        # Cluster rows, that are close to each other
        # Initialize cluster number and cluster list
        cluster_number = 1
        cluster_list = []
        times = time_attribution[date_col].values
        # Iterate through the DataFrame and cluster date-times
        for idx, row in enumerate(times):
            if idx == 0:
                cluster_list.append(cluster_number)
            else:
                time_diff = row - times[idx - 1]
                if time_diff <= timedelta_threshold:
                    cluster_list.append(cluster_number)
                else:
                    cluster_number += 1
                    cluster_list.append(cluster_number)

        time_attribution['cluster'] = cluster_list
        # Create a DataFrame from the filtered rows
        # Set the aggregated DataFrame to member
        return time_attribution

    def set_image_attribution(self, filename: str, type: InputImageMode = InputImageMode.All,
                              timedelta_threshold: pd.Timedelta = pd.Timedelta(240, 'm')) -> None:

        if type == self.InputImageMode.All:
            self.image_list = self.read_input_image_attribution(filename)

        elif type == self.InputImageMode.Daily:
            self.image_list = self.read_input_image_attribution_daily(filename)

        elif type == self.InputImageMode.Events:
            all = self.read_input_image_attribution(filename)
            dailies = self.read_input_image_attribution_daily(filename)
            self.image_list = self.cluster_images(all.drop(dailies.index.values.tolist()), timedelta_threshold)

    def read_camera_calibration(self, filename):
        self.calibration_list = pd.read_csv(filename, sep=',', header=None, names=['camera', 'camera_path'])
        logger.info(f'Read calibration list file: {filename} with shape: {self.calibration_list.shape}')

    def set_cameras(self, camera_list: list):
        self.cameras_list = camera_list

    def import_time_slices(self):

        # check, if member variables are set

        # extract calibration-path from dataframe
        calib_list = [(row[0], row[1]) for _, row in self.calibration_list.iterrows()]
        # iterate over rows and insert them into chunk

        for idx, image_row in tqdm(self.image_list.iterrows()):

            chunk = self.doc.addChunk()
            img_path = ''
            img_label = ''
            calib_path = ''

            for camera in self.cameras_list:
                for cell in image_row.items():
                    if f"{camera}_path" in cell[0]:
                        img_path = cell[1]

                    if f"{camera}_pic" in cell[0]:
                        img_label = cell[1]

                for _, calib_row in self.calibration_list.iterrows():
                    if camera in calib_row['camera']:
                        calib_path = calib_row['camera_path']

                if camera in self.image_list.columns[0]:
                    # print(type(image_row[f'{camera}_datetime']))
                    unix = pd.to_datetime(image_row[f'{camera}_datetime'], unit='s')
                    # print(unix, type(unix))
                    timestamp = unix.strftime('%Y-%m-%dT%H:%M:%S')

                    if "cluster" in self.image_list.columns:
                        # chunk.label = str(idx) + '_' + timestamp
                        chunk.label = f"{image_row['cluster']}_{idx}_{timestamp}"
                    else:
                        chunk.label = str(idx) + '_' + timestamp

                if not img_path or not calib_path:
                    continue  # skip

                group = chunk.addCameraGroup()
                group.label = camera

                sensor = chunk.addSensor()

                if os.path.exists(calib_path):
                    calib = Metashape.Calibration()
                    calib.load(calib_path)
                    sensor.user_calib = calib
                    sensor.fixed = False

                camera = chunk.addCamera(sensor)
                camera.open(img_path)
                camera.label = img_label
                camera.group = group

    def import_marker_coordinates(self, path, camname):
        self.marker.import_marker_from_dir(path, camname)
        logger.info(
            f'Imported marker coordinates from {path} for {camname}. New shape: {self.marker.marker_frame.shape}')

    def import_marker_label(self, filename):
        self.marker.import_marker_label_positions(filename)
        logger.info(f'Imported marker label positons from {filename}. New shape: {self.marker.marker_label.shape}')

    def import_marker_measurements(self, *chunks, refine_labels=[], dir_fit_result=''):

        for chunk in tqdm(chunks):
            with IOMeasurements(chunk, marker_positions=self.marker.marker_frame,
                                marker_label=self.marker.marker_label) as io_meas:
                io_meas.refine_ellipse_labels = refine_labels
                io_meas.refine_ellipse_output_dir = dir_fit_result
                io_meas.refine_ellipse_labels = refine_labels
                io_meas.import_marker_measurements()

    def clean_insufficient_marker(self, *chunks, min_num_projections=2):

        for chunk in tqdm(chunks):
            for idx, marker in enumerate(chunk.markers):
                if len(marker.projections.values()) < min_num_projections:
                    print(f'erased {marker.label}, because {len(marker.projections.values())}')
                    self.exclude_marker(chunk, marker_label=marker.label)

    def find_closest_point(self, points, target, dist_tresh):
        closest_point = None
        closest_distance = float('inf')
        close_enough = True
        for point in points:
            distance = ((float(point[2]) - float(target[0])) ** 2 + (float(point[3]) - float(target[1])) ** 2) ** 0.5
            if distance < closest_distance:
                closest_distance = distance
                closest_point = point

        if closest_distance > dist_tresh:
            close_enough = False

        return close_enough, closest_point

    def find_similar_point(self, label_set: list, marker_df: pd.DataFrame, query_row: pd.Series):
        """ Find points based on distance pattern to neighboring points. in work"""

        query_x = float(query_row['xc'])
        query_y = float(query_row['yc'])

        # calculate distance to each point for label in marker_df
        query_net = sorted([(math.dist([query_x, float(query_y)], [float(row['xc']), float(row['yc'])]),
                             math.atan2(float(row['yc'] - query_y), float(row['xc']) - query_x)) for idx, row in
                            marker_df.iterrows()])

        # iterate over each point in marker_label and calculate each combination of

        label = ''
        sum = float('inf')

        for item in label_set:
            temp_net = sorted([(math.dist([item[2], item[3]], [row[2], row[3]]),
                                math.atan2(row[3] - item[3], row[2] - item[2])) for row in label_set])

            diff = np.array([x[1] - x[0] for x in list(zip(query_net, temp_net))]).sum()

            if abs(diff) < sum:
                label = item[1]
                sum = diff

        return label, sum

    def align_chunk(self, *chunks, alignment_params=None, skip_aligned=False):

        for chunk in tqdm(chunks):

            if [camera for camera in chunk.cameras if camera.transform is not None] and skip_aligned:
                continue

            if alignment_params is None:
                alignment_params = self.align_params

            try:
                # feature tracking and matching in photos
                chunk.matchPhotos(downscale=self.align_params.downscale,
                                  generic_preselection=alignment_params.generic_preselection,
                                  reference_preselection=alignment_params.reference_preselection,
                                  filter_mask=alignment_params.filter_mask,
                                  mask_tiepoints=alignment_params.mask_tiepoints,
                                  keypoint_limit=alignment_params.keypoint_limit,
                                  tiepoint_limit=alignment_params.tiepoint_limit,
                                  reset_matches=alignment_params.reset_matches,
                                  guided_matching=alignment_params.guided_matching)

                # camera alignment based on tracked features
                chunk.alignCameras(reset_alignment=alignment_params.reset_alignment)

                # another alignment, if not all cameras are aligned
                if any([not bool(camera.transform) for camera in chunk.cameras]):
                    chunk.alignCameras(reset_alignment=False)

                # re-optimization with enabeled/disabled parameters of inner orientation
                chunk.optimizeCameras(fit_f=self.align_params.fit_f,
                                      fit_cx=self.align_params.fit_cx,
                                      fit_cy=self.align_params.fit_cy,
                                      fit_b1=self.align_params.fit_b1,
                                      fit_b2=self.align_params.fit_b2,
                                      fit_k1=self.align_params.fit_k1,
                                      fit_k2=self.align_params.fit_k2,
                                      fit_k3=self.align_params.fit_k3,
                                      fit_k4=self.align_params.fit_k4,
                                      fit_p1=self.align_params.fit_p1,
                                      fit_p2=self.align_params.fit_p2,
                                      fit_corrections=self.align_params.fit_corrections,
                                      adaptive_fitting=self.align_params.adaptive_fitting,
                                      tiepoint_covariance=self.align_params.tiepoint_covariance)

            except Exception as e:
                if hasattr(e, 'message'):
                    logger.error(f'---Alignment---: chunk {chunk.label} failed: {e.message}')
                else:
                    logger.error(f'---Alignment---: chunk {chunk.label} failed: {e}')
                continue

                chunk.optimizeCameras(fit_f=self.align_params.fit_f,
                                      fit_cx=self.align_params.fit_cx,
                                      fit_cy=self.align_params.fit_cy,
                                      fit_b1=self.align_params.fit_b1,
                                      fit_b2=self.align_params.fit_b2,
                                      fit_k1=self.align_params.fit_k1,
                                      fit_k2=self.align_params.fit_k2,
                                      fit_k3=self.align_params.fit_k3,
                                      fit_k4=self.align_params.fit_k4,
                                      fit_p1=self.align_params.fit_p1,
                                      fit_p2=self.align_params.fit_p2,
                                      fit_corrections=self.align_params.fit_corrections,
                                      adaptive_fitting=self.align_params.adaptive_fitting,
                                      tiepoint_covariance=self.align_params.tiepoint_covariance)

            except Exception as e:
                if hasattr(e, 'message'):
                    logger.error(f'---Alignment---: chunk {chunk.label} failed: {e.message}')
                else:
                    logger.error(f'---Alignment---: chunk {chunk.label} failed: {e}')
                continue

    def dense_chunk(self, *chunks, skip_bad_chunks=True, build_all=True, ignore_warning=True):

        for chunk in tqdm(chunks):
            if not ignore_warning:
                if 'WARNING' in chunk.label:
                    continue

            if not chunk.dense_cloud or build_all:
                try:
                    chunk.buildDepthMaps(downscale=1, filter_mode=Metashape.AggressiveFiltering)
                    chunk.buildDenseCloud()
                except Exception as e:
                    if hasattr(e, 'message'):
                        logger.error(f'---Dense Matching---: chunk {chunk.label} failed: {e.message}')
                    else:
                        logger.error(f'---Dense Matching---: chunk {chunk.label} failed: {e}')
                    continue

    def refine_alignment(self, *chunks, alignment_params=None, iterations=1):

        if alignment_params is None:
            alignment_params = copy.deepcopy(self.align_params)
            alignment_params.reset_params()

        errorMessage = []

        project_dir = os.path.dirname(os.path.abspath(self.project_path))

        # thresholds sparse filter
        threshold_reprojection = 0.5  # 0.5 It's a good first approach
        threshold_reconstruction = 25
        threshold_projection = 4

        iterationsAlignMike = 1  # 2, iterations alignment (Mike inspired)
        downScaleImagesStart = 1  # 0 highest, 1 high, 2 medium...
        key_pointLimitStart = 50000
        tie_pointLimitStart = 4000
        key_pointLimitIncrease = 2
        minimumNbrSparsePoints = 75
        reprojRMSE_thresh = 1  # in pixels
        guidedMatchingStart = True  # this will create more tie points but also take a lot longer
        accuracyTiepoints = 0.5
        accuracyMarker = 0.001  # in [mm]
        formatMarkerFile = 'nxyz'  # nxyz: no accuracy for each point given; nxyzXYZ: accuracy for each point given

        '''STEP 4 -> Align and optimization Cameras (Loop inspired by Mike James 2017)'''

        for index, chunk in tqdm(enumerate(chunks)):

            # prepare log file
            stdoutOrigin = sys.stdout
            if not os.path.isdir(rf"{project_dir}\logFiles"):
                os.mkdir(rf"{project_dir}\logFiles")

            sys.stdout = open(
                rf"{project_dir}\logFiles\log_metashape_{chunk.label.replace(':', '-').replace('[WARNING]_', '')}.log",
                "w")

            try:
                chunk.tiepoint_accuracy = accuracyTiepoints
                for i in range(iterationsAlignMike):
                    alignment_params.keypoint_limit = key_pointLimitStart
                    alignment_params.tiepoint_limit = tie_pointLimitStart
                    alignment_params.downscale = downScaleImagesStart
                    alignment_params.guided_matching = guidedMatchingStart
                    Metashape.app.update()
                    t0 = time.time()
                    # repeat matching photos with increased number keypoints until at least minimum number of sparse points found
                    # (and repeat matching only during first of Mike's iteration)
                    iterations = 4
                    for l in range(iterations):
                        if i > 0:
                            l = iterations - iterationsAlignMike
                        if not l == 0:
                            # assignInteriorCameraGeometry(chunk_original, uniqueCameras, sensorInfos, errorMessage, index, True)
                            pass

                        reprojection_rmse, reprojection_maxError = self.calc_reprojection(chunk)
                        self.align_chunk(chunk, alignment_params=alignment_params)
                        reprojection_rmse, reprojection_maxError = self.calc_reprojection(chunk)

                        if chunk.point_cloud.points:
                            reprojection_rmse, reprojection_maxError = self.calc_reprojection(chunk)
                            print(
                                rf"Match iteration:  + {str(l)} + ; number of sparse points:  + {str(len([p for p in chunk.point_cloud.points]))}")
                            # continue only if sufficient number of matched points and accurancy reconstruction high enough (RMSE below threshold)
                            if len([p for p in
                                    chunk.point_cloud.points]) > minimumNbrSparsePoints and reprojection_rmse < reprojRMSE_thresh:
                                break
                            if len([p for p in
                                    chunk.point_cloud.points]) < minimumNbrSparsePoints and guidedMatching == False:
                                guidedMatching = True
                                continue
                            if reprojection_rmse > reprojRMSE_thresh and int(alignment_params.downscale) > 0:
                                alignment_params.downscale = int(alignment_params.downscale - 1)
                                print(f"re-iterated due to reprojection error of {str(reprojection_rmse)}")
                                continue
                            if len([p for p in chunk.point_cloud.points]) < minimumNbrSparsePoints:
                                alignment_params.keypoint_limit = int(
                                    alignment_params.keypoint_limit * key_pointLimitIncrease)
                                alignment_params.tiepoint_limit = int(
                                    alignment_params.tiepoint_limit * key_pointLimitIncrease / 2)
                                print(
                                    f"re-iterated due to too low number of matched points and thus key points increased to {str(alignment_params.keypoint_limit)}")
                        else:
                            print(f"Match iteration: {str(l)}; no matches")
                            if alignment_params.guided_matching == False:
                                alignment_params.guided_matching = True
                            else:
                                alignment_params.keypoint_limit = int(
                                    alignment_params.keypoint_limit * key_pointLimitIncrease)
                                alignment_params.tiepoint_limit = int(
                                    alignment_params.tiepoint_limit * key_pointLimitIncrease / 2)
                    # doc.save(path_project + name_project + ".psx")
                    print("Camera alignment performed")
                    # reset, which camera parameters are fixed
                    try:
                        for camera in chunk.cameras:
                            camera.sensor.fixed_params = ["K1", "K2", "K3", "K4", "P1", "P2", "B1", "B2"]
                            # sensorCam.fixed = True
                    except:
                        print(traceback.format_exc())
                        print("Error while (re-)setting camera calibration")
                    t1 = time.time()
                    chunk.optimizeCameras(fit_corrections=False, adaptive_fitting=False, tiepoint_covariance=False)
                    chunk.updateTransform()
                    '''STEP 5 -> Compute RMS reprojection Error (AGISOFT FORUM)'''
                    try:
                        reprojection_rmse, reprojection_maxError = self.calc_reprojection(chunk)
                        print("Average tie point residual error: " + str(reprojection_rmse))
                        print("Maximum tie point residual error: " + str(reprojection_maxError))
                        '''STEP 6 -> Compute Markers error (AGISOFT FORUM)'''
                        error_pix_GCP, error_pix_CP, error_GCP_3D, error_CP_3D = self.calc_reprojectionMarker(chunk)
                        print("Marker point residual error (GCP): " + str(error_pix_GCP))
                        print("Marker point residual error (CP): " + str(error_pix_CP))
                        print("Marker point residual error 3D (GCP): " + str(error_GCP_3D))
                        print("Marker point residual error 3D (CP): " + str(error_CP_3D))
                        '''STEP 7 -> Update accuracies and loop again alignment (Loop inspired by Mike James 2017)'''
                        if iterationsAlignMike > 1:
                            chunk.tiepoint_accuracy = reprojection_rmse
                    except Exception as e:
                        print(e)
                        print('failed RMSE calculation')
                    # chunk_original.marker_projection_accuracy = error_pix #not updated here because measured with ellipse fit with known accuracy
                # stop processing if reprojection still too high
                if reprojection_rmse > reprojRMSE_thresh:
                    # close log file
                    sys.stdout.close()
                    sys.stdout = stdoutOrigin
                    print("processing aborted due to too high reprojection error")
                    continue
            except:
                print(traceback.format_exc())
                errorMessage.append([index, 'alignment failed'])
                errorMessageSave = pd.DataFrame(errorMessage)
                errorMessageSave.to_csv(rf"{project_dir}\successMS.txt")
                continue
            '''STEP 8 -> Point Filter (based on reprojection error)'''
            try:
                # filter sparse point cloud
                f = Metashape.PointCloud.Filter()
                f.init(chunk, criterion=Metashape.PointCloud.Filter.ReprojectionError)
                f.removePoints(threshold_reprojection)
                f = Metashape.PointCloud.Filter()
                f.init(chunk, criterion=Metashape.PointCloud.Filter.ReconstructionUncertainty)
                f.removePoints(threshold_reconstruction)
                f = Metashape.PointCloud.Filter()
                f.init(chunk, criterion=Metashape.PointCloud.Filter.ProjectionAccuracy)
                f.removePoints(threshold_projection)
            except:
                print(traceback.format_exc())
                errorMessage.append([index, 'sparse point cloud filtering failed'])
                errorMessageSave = pd.DataFrame(errorMessage)
                errorMessageSave.to_csv(rf"{project_dir}\successMS.txt")
            '''STEP 8 -> Cameras optimization after last loop iteration and point filter. Tie Point Covariance exported!'''
            try:
                chunk.optimizeCameras(fit_corrections=False, adaptive_fitting=False, tiepoint_covariance=True)
                # Update Transform
                chunk.updateTransform()
                reprojection_rmse, reprojection_maxError = self.calc_reprojection(chunk)
                if reprojection_rmse > reprojRMSE_thresh:
                    # close log file
                    sys.stdout.close()
                    sys.stdout = stdoutOrigin
                    continue
            except:
                print(' ')
            '''STEP 9 -> Export tie point covariance'''
            try:
                if not os.path.isdir(rf"{project_dir}\ptPrecision"):
                    os.mkdir(rf"{project_dir}\ptPrecision")

                self.exportTiepointCovariance(chunk,
                                              fr"{project_dir}\ptPrecision\pt_prec_{chunk.label.replace(':', '-').replace('[WARNING]_', '')}.txt")
                self.save_project()
            except:
                print(traceback.format_exc())
                errorMessage.append([index, 'failed getting tie point covariance'])
                errorMessageSave = pd.DataFrame(errorMessage)
                errorMessageSave.to_csv(rf"{project_dir}\successMS.txt")

    def mark_error_chunks(self, *chunks, threshold_2d=1.0, threshold_3d=0.005):

        count_errors = 0

        for chunk in tqdm(chunks):
            try:
                # sigma, maxe = self.calc_reprojection(chunk)
                error_pix_GCP, error_pix_CP, error_GCP_3D, error_CP_3D = self.calc_reprojectionMarker(chunk)

                if any([bool(x > threshold_2d) for x in [error_pix_GCP, error_pix_CP]]) or any(
                        [bool(x > threshold_3d) for x in [error_GCP_3D, error_CP_3D]]):
                    # mark chunk, inspec it manually
                    count_errors += 1
                    if not 'WARNING' in chunk.label:
                        chunk.label = f'[WARNING]_{chunk.label}'

                else:
                    chunk.label = chunk.label.replace('[WARNING]_', '')

            except Exception as e:
                if hasattr(e, 'message'):
                    logger.error(f'---Mark Error Chunks---: chunk {chunk.label} failed: {e.message}')
                else:
                    logger.error(f'---Mark Error Chunks---: chunk {chunk.label} failed: {e}')
                continue

        logger.info(f'Counted Errors in workspace: {count_errors}')

    def calc_reprojection(self, chunk):
        """@authors: Anette Eltner (based on scripts by XBG, xabierblanch@gmail.com)"""
        point_cloud = chunk.point_cloud
        points = point_cloud.points
        npoints = len(points)
        projections = chunk.point_cloud.projections
        err_sum = 0
        num = 0
        maxe = 0

        if npoints == 0:
            print('no sparse points')
            return

        point_ids = [-1] * len(point_cloud.tracks)
        point_errors = dict()
        for point_id in range(0, npoints):
            point_ids[points[point_id].track_id] = point_id

        for camera in chunk.cameras:
            if not camera.transform:
                continue
            for proj in projections[camera]:
                track_id = proj.track_id
                point_id = point_ids[track_id]
                if point_id < 0:
                    continue
                point = points[point_id]
                if not point.valid:
                    continue
                error = camera.error(point.coord, proj.coord).norm() ** 2
                err_sum += error
                num += 1
                if point_id not in point_errors.keys():
                    point_errors[point_id] = [error]
                else:
                    point_errors[point_id].append(error)
                if math.sqrt(error) > maxe: maxe = math.sqrt(error)

        sigma = math.sqrt(err_sum / num)
        return sigma, maxe  # point_errors

    def calc_reprojectionMarker(self, chunk):
        """@authors: Anette Eltner (based on scripts by XBG, xabierblanch@gmail.com)"""
        rmsGCP = []
        rmsCP = []
        diff3D_GCP = []
        diff3D_CP = []
        numSqGCP = 0
        numSqCP = 0
        num3DGCP = 0
        num3DCP = 0
        for marker in chunk.markers:
            countFailedGCP = 0
            try:
                # error in image space
                diff = []
                for camera in chunk.cameras:
                    # T = camera.transform.inv()
                    # calib = camera.sensor.calibration
                    try:
                        v_proj = marker.projections[
                            camera].coord  # 2 dimensional vector of the marker projection on the photo
                        v_reproj = camera.project(
                            marker.position)  # 2 dimensional vector of projected 3D marker position
                        error = (v_proj - v_reproj).norm()  # reprojection error for current photo
                        diff.append(error)
                        if marker.reference.enabled:
                            rmsGCP.append(error ** 2)
                            numSqGCP += 1
                        else:
                            rmsCP.append(error ** 2)
                            numSqCP += 1
                    except Exception as e:
                        # print(e)
                        # print('marker ' + marker.label + ' in camera ' + camera.label + ' has no projection')
                        countFailedGCP = countFailedGCP + 1

                if countFailedGCP != 0:
                    print('marker ' + marker.label + ' failed in ' + str(countFailedGCP) + ' images')

                # error in object space
                error = (chunk.transform.matrix.mulp(marker.position) - chunk.crs.unproject(
                    marker.reference.location)).norm() ** 2
                if marker.reference.enabled:
                    diff3D_GCP.append(error)
                    num3DGCP += 1
                else:
                    diff3D_CP.append(error)
                    num3DCP += 1

            except:
                print(traceback.format_exc())
                print('marker ' + marker.label + ' failed in RMSE assessment')

        # image space
        rmsGCP = math.sqrt(sum(rmsGCP) / numSqGCP)
        rmsCP = math.sqrt(sum(rmsCP) / numSqCP)

        # object space
        rmsGCP_3D = math.sqrt(sum(diff3D_GCP) / num3DGCP)
        rmsCP_3D = math.sqrt(sum(diff3D_CP) / num3DCP)

        return rmsGCP, rmsCP, rmsGCP_3D, rmsCP_3D

    def exportTiepointCovariance(self, chunk, outFile):
        # Get transforms to account for real-world coordinate system (CRS)
        # Note, this resets the region to the default
        M = chunk.transform.matrix
        T = chunk.crs.localframe(M.mulp(chunk.region.center)) * M
        if chunk.transform.scale:
            R = chunk.transform.scale * T.rotation()
        else:
            R = T.rotation()

        # Open the output file and write the precision estimates to file
        with open(outFile, "w") as fid:
            # Open the output file
            fwriter = csv.writer(fid, delimiter='\t', lineterminator='\n')

            # Write the header line
            fwriter.writerow(['X(m)', 'Y(m)', 'Z(m)', 'sX(mm)', 'sY(mm)', 'sZ(mm)',
                              'covXX(m2)', 'covXY(m2)', 'covXZ(m2)', 'covYY(m2)', 'covYZ(m2)', 'covZZ(m2)'])

            # Iterate through all valid points, writing a line to the file for each point
            for point in chunk.point_cloud.points:
                if not point.valid:
                    continue

                # Transform the point coordinates into the output local coordinate system
                if chunk.crs:
                    V = M * (point.coord)
                    V.size = 3
                    pt_coord = chunk.crs.project(V)
                else:
                    V = M * (point.coord)
                    V.size = 3
                    pt_coord = V

                # Transform the point covariance matrix into the output local coordinate system
                pt_covars = R * point.cov * R.t()

                # Write the line of coordinates, precisions and covariances to the text file
                fwriter.writerow([
                    '{0:0.5f}'.format(pt_coord[0]), '{0:0.5f}'.format(pt_coord[1]), '{0:0.5f}'.format(pt_coord[2]),
                    '{0:0.7f}'.format(math.sqrt(pt_covars[0, 0]) * 1000),
                    '{0:0.7f}'.format(math.sqrt(pt_covars[1, 1]) * 1000),
                    '{0:0.7f}'.format(math.sqrt(pt_covars[2, 2]) * 1000),
                    '{0:0.9f}'.format(pt_covars[0, 0]), '{0:0.9f}'.format(pt_covars[0, 1]),
                    '{0:0.9f}'.format(pt_covars[0, 2]),
                    '{0:0.9f}'.format(pt_covars[1, 1]), '{0:0.9f}'.format(pt_covars[1, 2]),
                    '{0:0.9f}'.format(pt_covars[2, 2])])

            # Close the text file
            fid.close()

    def exclude_marker(self, *chunks, marker_label):

        for chunk in tqdm(chunks):
            for idx, marker in enumerate(chunk.markers):
                if marker_label in marker.label:
                    try:
                        chunk.remove(chunk.markers[idx])
                    except IndexError as e:
                        print(e)
                        continue

    def import_marker_reference(self, *chunks, reference_file, columns='nxyz', delimiter=',',
                                marker_accuracy=[0.001, 0.001, 0.001]):

        for chunk in tqdm(chunks):
            chunk.importReference(reference_file, format=Metashape.ReferenceFormatCSV, columns=columns,
                                  delimiter=delimiter, items=Metashape.ReferenceItemsMarkers,
                                  crs=Metashape.CoordinateSystem(
                                      'LOCAL_CS["Local CS",LOCAL_DATUM["Local Datum",0],UNIT["metre",1]]'),
                                  create_markers=False)

            for marker in chunk.markers:
                marker.reference.accuracy = marker_accuracy

    def set_check_marker(self, *chunks, check_marker_label: list):

        for chunk in chunks:

            for marker in chunk.markers:
                marker.enabled = True
                marker.reference.enabled = True

                # print(map(marker.label.__contains__, check_marker_label))

                if any(map(marker.label.__contains__, check_marker_label)):
                    marker.reference.enabled = False

    def plot_marker_accuracies(self, *chunks):

        # class for storing error on marker
        class GCPMarkerError:
            def __init__(self, marker_label):
                self.marker_label: str = marker_label
                self.control_error: list((int, float)) = []
                self.check_error: list((int, float)) = []

        marker_objects = []
        for idx, chunk in enumerate(chunks):
            for marker in chunk.markers:
                marker_obj = [x for x in marker_objects if x.marker_label in marker.label]

                if not marker_obj:
                    marker_obj = GCPMarkerError(marker.label)
                    marker_objects.append(marker_obj)

                if isinstance(marker_obj, list):
                    marker_obj = marker_obj[0]

                try:
                    source = marker.reference.location
                    estim = chunk.crs.project(chunk.transform.matrix.mulp(marker.position))
                    if source and estim:
                        error = estim - source
                        total = error.norm()

                        if marker.reference.enabled:
                            marker_obj.control_error.append((idx, total))
                        else:
                            marker_obj.check_error.append((idx, total))

                except:
                    continue

        # plot marker accuracies in subplots:

        subplot_cols = 2
        subplot_rows = math.ceil((len(marker_objects) + 1) / subplot_cols)
        plt.style.use('seaborn')

        fig, axs = plt.subplots(subplot_rows, subplot_cols, sharex=True, sharey=False, figsize=(12, 12))
        plt.subplots_adjust(hspace=0.3, wspace=0.1)
        # id_last_element = len(marker_objects) - 1
        for id, mrk_obj in enumerate(marker_objects):
            plot_row = id // subplot_cols
            plot_col = id % subplot_cols

            axs[plot_row, plot_col].plot([x[0] for x in mrk_obj.control_error], [x[1] for x in mrk_obj.control_error],
                                         label='Error Controlpoints [m]')
            axs[plot_row, plot_col].plot([x[0] for x in mrk_obj.check_error], [x[1] for x in mrk_obj.check_error],
                                         label='Error Checkpoints [m]')
            axs[plot_row, plot_col].set_title(f'Marker {mrk_obj.marker_label}')
            axs[plot_row, plot_col].set_ylim([0, 0.02])

            if plot_col != 0:
                axs[plot_row, plot_col].set_yticklabels("")

            # if plot_row != subplot_rows:
            #     axs[plot_row, plot_col].set_xticklabels("")

            # if id == id_last_element:
            #     axs[plot_row, plot_col].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
            # axs[plot_row, plot_col].set_yscale('log')

        # calculate total gcp error

        control = list(itertools.chain.from_iterable([x.control_error for x in marker_objects if x.control_error]))
        check = list(itertools.chain.from_iterable([x.check_error for x in marker_objects if x.check_error]))

        control_temps = []
        check_temps = []
        for i in range(len(chunks)):
            control_temps.append([i, [x[1] for x in control if i == x[0]]])
            check_temps.append([i, [x[1] for x in check if i == x[0]]])

        control_mean = []
        check_mean = []
        # calculate mean
        for i in range(len(chunks)):
            # control_mean = [(x[0], statistics.fmean(x[1])) for x in control_temps]
            control_mean = [(x[0], np.mean(x[1])) for x in control_temps]
            check_mean = [(x[0], np.mean(x[1])) for x in check_temps]

        plot_row = len(marker_objects) // subplot_cols
        plot_col = len(marker_objects) % subplot_cols

        axs[plot_row, plot_col].plot([x[0] for x in control_mean], [x[1] for x in control_mean],
                                     label='Error Controlpoints [m]')
        axs[plot_row, plot_col].plot([x[0] for x in check_temps], [x[1] for x in check_temps],
                                     label='Error Checkpoints [m]')
        axs[plot_row, plot_col].set_title(f'Total Mean Error')
        axs[plot_row, plot_col].set_ylim([0, 0.02])
        axs[plot_row, plot_col].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))

        if plot_col != 0:
            axs[plot_row, plot_col].set_yticklabels("")

        # print(append_control)
        # fig.show()

    def export_dense_clouds(self, *chunks):

        project_dir = os.path.dirname(os.path.abspath(self.project_path))
        if not os.path.isdir(rf"{project_dir}\Dense"):
            os.mkdir(rf"{project_dir}\Dense")

        for chunk in tqdm(chunks):
            if chunk.dense_cloud:
                # Set save name
                chunk_label = chunk.label.replace(':', '-').replace('[WARNING]_', '')
                # if chunk label contains event cluster number in front, split it off
                if re.search(r'\b\d{1,4}_\d{1,4}_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', chunk_label):
                    chunk_label = chunk_label.split("_", 1)[1]

                # export_chunk_label = fr"{project_dir}\Dense\{chunk.label.replace(':','-')}.ply".replace('[WARNING]_', '')
                export_chunk_label = os.path.join(project_dir, "Dense",
                                                  f"{chunk_label}.ply")

                chunk.exportPoints(export_chunk_label, source_data=Metashape.DenseCloudData,
                                   format=Metashape.PointsFormatPLY, binary=True, save_colors=True)

    def calculate_marker_reprojection_error(self, *chunks):
        """Calculate each 2D Projection and it's error of each marker in each photo"""

        for chunk in chunks:

            marker_reproj_error = []

            for marker in chunk.markers:
                if not marker.position:
                    print("{marker.label} is not defined in 3D, skipping...")
                    continue
                position = marker.position
                proj_error = []
                proj_sqsum = 0

                for camera in marker.projections.keys():
                    if not camera.transform:
                        continue  # skipping not aligned cameras
                    image_name = camera.label
                    proj = marker.projections[camera].coord
                    reproj = camera.project(marker.position)
                    error = reproj - proj

                    # print(f'{marker.label} - {camera.group} - {error.norm()}')

                    marker_reproj_error.append([camera, marker, error.norm()])

                    proj_error.append(error.norm())
                    proj_sqsum += error.norm() ** 2

                if len(proj_error):
                    # 3sigma on reproj error

                    repr_error = np.array([x[2] for x in marker_reproj_error])
                    mean = np.mean(repr_error, axis=0)

                    error = math.sqrt(proj_sqsum / len(proj_error))
                    # print(f"{marker.label} projection error: {mean}")

            return marker_reproj_error

    def delete_marker_with_largest_error(self, *chunks, threshold=1, align_after=True, iterate=True):

        for chunk in chunks:
            # marker_error = self.calculate_marker_reprojection_error(chunk)

            error_marker = [x for x in self.calculate_marker_reprojection_error(chunk) if (x[2] > threshold)]

            if len(error_marker):
                max_error_element = max(error_marker, key=lambda x: x[2])
                # delete marker on image

                max_error_element[1].projections[max_error_element[0]] = None
                logger.info(
                    f'Deleted Projection for {max_error_element[0].label} on {max_error_element[1].label} with error {max_error_element[2]}')
                if align_after:
                    self.align_chunk(chunk)
                    if iterate:
                        self.delete_marker_with_largest_error(chunk, threshold=threshold, align_after=align_after,
                                                              iterate=iterate)

    def disable_bad_images(self, *chunks: list[Metashape.Chunk], img_threshold: float = 0.3):

        for chunk in tqdm(chunks):
            chunk.analyzePhotos(chunk.cameras)

            for camera in chunk.cameras:
                quality = camera.frames[0].meta["Image/Quality"]
                if float(quality) < img_threshold:
                    chunk.remove(camera)


class MetashapeAlignmentParameters:

    def __init__(self):
        self.downscale = 1
        self.generic_preselection = False
        self.reference_preselection = False
        self.filter_mask = False
        self.mask_tiepoints = False
        self.keypoint_limit = 40000
        self.tiepoint_limit = 4000
        self.reset_matches = True
        self.guided_matching = True
        self.reset_alignment = True

        self.fit_f = True
        self.fit_cx = True
        self.fit_cy = True
        self.fit_b1 = False
        self.fit_b2 = False
        self.fit_k1 = False
        self.fit_k2 = False
        self.fit_k3 = False
        self.fit_k4 = False
        self.fit_p1 = False
        self.fit_p2 = False
        self.fit_corrections = False
        self.adaptive_fitting = False
        self.tiepoint_covariance = False

        self.fit_f = True
        self.fit_cx = True
        self.fit_cy = True
        self.fit_b1 = False
        self.fit_b2 = False
        self.fit_k1 = False
        self.fit_k2 = False
        self.fit_k3 = False
        self.fit_k4 = False
        self.fit_p1 = False
        self.fit_p2 = False
        self.fit_corrections = False
        self.adaptive_fitting = False
        self.tiepoint_covariance = False

    def reset_params(self):
        self.downscale = 1
        self.generic_preselection = False
        self.reference_preselection = False
        self.filter_mask = False
        self.mask_tiepoints = False
        self.keypoint_limit = 40000
        self.tiepoint_limit = 4000
        self.reset_matches = True
        self.guided_matching = True
        self.reset_alignment = True

        self.fit_f = True
        self.fit_cx = True
        self.fit_cy = True
        self.fit_b1 = False
        self.fit_b2 = False
        self.fit_k1 = False
        self.fit_k2 = False
        self.fit_k3 = False
        self.fit_k4 = False
        self.fit_p1 = False
        self.fit_p2 = False
        self.fit_corrections = False
        self.adaptive_fitting = False
        self.tiepoint_covariance = False



if __name__ == '__main__':
    metashape_path = r"D:\Grothum\Kamerstation_Metashape\Station_unten\2022-05\test.psx"
    attri_file = r"D:\Grothum\Kamerastationen\Station_oben\Time_Attribution_Station_oben_2022-05.txt"

    proc = MetashapeProcessing()
    proc.project_path = metashape_path
    proc.setup_project()
    proc.reset_project()
    proc.set_image_attribution(attri_file, proc.InputImageMode.Daily)
    proc.read_camera_calibration(r'D:\Grothum\Kamerastationen\calibration_paths.txt')
    cam_list = ['Kamera21', 'Kamera22', 'Kamera23', 'Kamera24', 'Kamera25']
    proc.set_cameras(cam_list)
    proc.import_time_slices()
    proc.save_project()
