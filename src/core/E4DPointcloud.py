from __future__ import annotations

import logging
import os
import sys
import subprocess
from dataclasses import dataclass
from enum import Enum
import chardet

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

cloudcompy_cc_path = r'D:\Grothum\source\CloudComPy310_20230705\CloudComPy310\CloudCompare'
if cloudcompy_cc_path not in sys.path:
    sys.path.append(cloudcompy_cc_path)
import cloudComPy as cc
os.environ["_CCTRACE_"] = "ON"  # set debug trace markers on
if cc.isPluginM3C2():
    import cloudComPy.M3C2

# logging

logger = logging.getLogger()
fhandler = logging.FileHandler(filename='pointcloud_processing_log.log', mode='a')
logger.addHandler(fhandler)
formatter = logging.Formatter('[%(asctime)s %(message)s]')
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)
fhandler.setFormatter(formatter)
logger.setLevel(logging.INFO)


@dataclass
class NoiseFilterParameter:
    kernel_radius = 1.0
    knn = 6
    sigma = 1.0
    remove_isolated_points = True
    use_knn = False
    use_absolute_error = False
    absolute_error = 1.0


@dataclass
class RasterGeotiffParameter:
    grid_step: float = 0.005
    vertDir: cc.CC_DIRECTION = cc.CC_DIRECTION.Z
    outputRasterZ: bool = True
    outputRasterSFs: bool = True
    outputRasterRGB: bool = True
    pathToImages: str = '.'
    resample: bool = False
    projectionType: cc.ProjectionType = cc.ProjectionType.PROJ_AVERAGE_VALUE
    sfProjectionType: cc.ProjectionType = cc.ProjectionType.PROJ_AVERAGE_VALUE
    emptyCellFillStrategy: cc.EmptyCellFillOption = cloudComPy.EmptyCellFillOption.LEAVE_EMPTY
    DelaunayMaxEdgeLength: float = 1.0
    KrigingParamsKNN: int = 8
    customHeight: float = float('nan')
    percentile: float = 50
    export_perCellCount: bool = False
    export_perCellMinHeight: bool = False
    export_perCellMaxHeight: bool = False
    export_perCellAvgHeight: bool = False
    export_perCellHeightStdDev: bool = False
    export_perCellHeightRange: bool = False
    export_perCellMedian: bool = False
    export_perCellPercentile: bool = False
    export_perCellUniqueCount: bool = False


class E4DPointcloud:

    def __init__(self):
        self.path: str = ""
        self.pointcloud: cc.ccPointCloud = cc.ccPointCloud()
        self.transform_matrix = cloudComPy.ccGLMatrix()
        self.m3c2_processing_params_file: str = None
        self.noise_filter_params: NoiseFilterParameter = NoiseFilterParameter()

    class CloudFileStructure(Enum):
        NONE = 1
        XYZ = 2
        XYZN = 3
        XYZNRGB = 4
        XYZS = 5
        XYZRGBS = 6

    def load_pointcloud(self, filename: str, file_structure: CloudFileStructure = CloudFileStructure.NONE,
                        delimiter=','):
        """
        Load pointcloud from file
        :param filename: filepath to pointcloud file
        :param file_structure: structure of file in CloudFileStructure
        :param delimiter: seperation delimiter in text files
        :return: self
        """
        self.path = filename
        if file_structure == E4DPointcloud.CloudFileStructure.NONE:
            self.pointcloud = cc.loadPointCloud(filename)
        elif file_structure == E4DPointcloud.CloudFileStructure.XYZS:
            with open(filename, 'rb') as rawdata:
                result = chardet.detect(rawdata.read(100000))
            file_array = pd.read_csv(filename, header=0, skiprows=0, delimiter=delimiter)
            self.pointcloud = self._ndarray_to_cloud(file_array.to_numpy(),
                                                     file_structure=file_structure,
                                                     column_names=list(file_array.columns))
            self.pointcloud.setName(os.path.basename(filename))

        logging.info(rf'Read Pointcloud from {filename} with {self.pointcloud.size()} Points')

        return self

    def write_pointcloud(self, filename: str):
        """
        Write pointcloud to given path
        :param filename: filename for export
        :return:
        """
        ret = cc.SavePointCloud(self.pointcloud, filename)


    def fit_plane(self):
        """
        Fit plane into pointcloud (least squares adjustment)
        :return:
        """
        return cloudComPy.ccPlane.Fit(self.pointcloud)

    def apply_transformation(self, transform: cloudComPy.ccGLMatrix):
        """
        Apply transformation to pointcloud
        :param transform: 4x4 transformation matrix
        :return:
        """
        self.transform_matrix *= transform
        self.pointcloud.applyRigidTransformation(transform)

        return self

    def remove_slope(self, save_transform_path=None):
        """
        Fits plane to pointcloud, then transforms point cloud so that the cloud and the estimated plane are
        parallel to xy-plane (normal is parallel to z-axis)
        :param save_transform_path:
        :return:
        """
        # fit plane
        plane = self.fit_plane()
        # return normal of plane and
        normal = plane.getNormal()
        z_axis = (0.0, 0.0, 1.0)

        transform = cloudComPy.ccGLMatrix.FromToRotation(normal, z_axis)
        self.apply_transformation(transform)

        if save_transform_path:
            self.save_transformation(transform, filename=save_transform_path)

        return self

    def remove_scalar_fields(self, field_names: list[str]):
        """
        Removes scalar fields of point cloud by given names
        :param field_names: names of scalar fields to remove
        :return:
        """
        for name in field_names:
            field_dict = self.pointcloud.getScalarFieldDic()
            scalar_id = field_dict.get(name, None)

            if scalar_id is not None:
                self.pointcloud.deleteScalarField(scalar_id)
            else:
                print(f'{name} not found as scalar field')

        return self

    def transform_scalar_fields(self, field_names: list[str], transform: cc.ccGLMatrix):
        """
        Geometric transformation of scalar fields by given 4x4 transformation matrix.
        Only Applicable on 3D-cartesian values
        :param field_names: names of scalar fields to transform
        :param transform: 4x4 transformation matrix
        :return:
        """
        arr, cols = self.cloud_to_ndarray()
        # get indices of columns given by field_names, clean out field_names if not existent
        clean_field_names = [substring for substring in field_names if substring in cols]
        # idx = [cols.index(substring) for substring in field_names if substring in cols]
        if not clean_field_names:
            raise ValueError("Could not find any field name in Scalar Field")
        # numpy array with 4 x n dimension of whole pointcloud
        fields = [self.pointcloud.getScalarField(x).toNpArrayCopy().reshape(-1, 1) for x in clean_field_names]
        dat = np.concatenate((*fields, np.ones(fields[0].shape)), axis=1)
        # transformation
        mat = np.array(transform.data()).reshape(4, 4)
        mat = np.linalg.inv(mat)
        dat = np.matmul(mat, dat.T)
        dat = np.delete(dat, -1, axis=0)
        # write back into scalar fields
        for name, col in zip(clean_field_names, dat):
            # print(name, col)
            self.pointcloud.getScalarField(name).fromNpArrayCopy(col)

        return self


    def crop_pointcloud(self, dim: int, points_array: np.ndarray = None, points_filename: str = None,
                        inside: bool = True):
        """
        Crop pointcloud by projecting points on plane specified by dim. Imports points, file points_filename is
        specified, else it uses points given in points_array.
        :param dim: dimension of cut plane: 0-XY, 1- YZ, 2-ZX
        :param points_array: numpy array (nx2) of point coordinates
        :param points_filename: filename to polyline (2 columns, with header, seperator = ','
        :param inside: if True, points inside are kept, False, points outside are kept
        :return:
        """
        if not points_array and not points_filename:
            logging.info(rf'No points or filename specified to crop pointcloud {self.pointcloud.getName()}, abort.')
            return

        if points_filename:
            df = pd.read_csv(points_filename, header=0, delimiter=',')
            points = df.iloc[:, 0:3].values
        elif points_array:
            points = points_array

        _, crop_line = self._ndarray_to_polyline(points)

        crop_line.setClosed(True)
        self.pointcloud = self.pointcloud.crop2D(crop_line, dim, inside)

        logging.info(rf'Crop {self.pointcloud.getName()} down to {self.pointcloud.size()} Points')

        return self

    def raster_pointcloud(self, grid_step: float, vert_dir=cc.CC_DIRECTION.Z):
        """
        Rasterize pointcloud in an evenly distributed grid.
        :param grid_step: distance between two points in length and width
        :param vert_dir: direction of interpolation.
        :return:
        """
        self.pointcloud = cc.RasterizeToCloud(self.pointcloud, gridStep=grid_step, vertDir=vert_dir)

        logging.info(rf'Raster {self.pointcloud.getName()} with {grid_step} down to {self.pointcloud.size()} Points')

    def interpolate_scalar(self, src_cloud: cc.ccPointCloud, sf_indexes: list, params: cc.interpolatorParameters):
        """

        :param src_cloud:
        :param sf_indexes:
        :param params:
        :return:
        """
        # TODO: Rename interpolated scalarfields after source scalar field
        cc.interpolateScalarFieldsFrom(self.pointcloud, src_cloud, sf_indexes, params)
        logging.info(rf'Interpolated {self.pointcloud.getName()} from {src_cloud.getName()} with '
                     rf'scalar fields: {sf_indexes}')

    def spatial_subsampling(self, minDistance=0.002, modParams=cloudComPy.SFModulationParams(), octree=None):
        """
        Resample cloud so that no points are less than minDistance away from each other
        :param minDistance: minimal distance of points beeing apart from each other
        :param modParams: modulation parameters of scalar fields
        :param octree: associate octree, if available
        :return:
        """
        ref = cloudComPy.CloudSamplingTools.resampleCloudSpatially(self.pointcloud, minDistance=minDistance,
                                                                   modParams=modParams, octree=octree)
        self.pointcloud = self.pointcloud.partialClone(ref)[0]

        return self

    def run_dbscan(self, eps=0.01, min_samples=5):
        """
        Run DBSCAN clustering of point cloud with given parameters. Appends dbscan labels as additional scalarfield
        'dbscan_label'
        :param eps:
        :param min_samples:
        :return:
        """

        df = self.cloud_to_dataframe()

        dbscan = DBSCAN(eps=3, min_samples=2).fit(df.iloc[:, :3])
        df['dbscan_label'] = dbscan.labels_

        # set df to dataframe
        if any(ele in df.columns for ele in ['R', 'G', 'B']):
            self.pointcloud = self._ndarray_to_cloud(df.values, file_structure=self.CloudFileStructure.XYZRGBS,
                                                     column_names=df.columns)
        else:
            self.pointcloud = self._ndarray_to_cloud(df.values, file_structure=self.CloudFileStructure.XYZS,
                                                     column_names=df.columns)


        def calculate_distance_to_plane(lr: LinearRegression, points: pd.DataFrame):
            """
            Calculate distance to plane
            :param lr: linear regression object created with sklearn
            :param points: points,
            :return:
            """
            # calculate difference from planepoint to actual point
            diff = points.iloc[:, 2:3] - lr.predict(points.iloc[:, :2])

            return diff.to_numpy()

        def weight_function(distances: np.ndarray):
            return np.reciprocal(np.abs(distances)).ravel()

        def std_deviation(distances: np.ndarray):
            return distances.std()

        result_dfs = []
        for x_start in np.arange(0, cloud['X'].max() + 1, tile_length):
            for y_start in np.arange(0, cloud['Y'].max() + 1, tile_length):
                tile = cloud[
                    (cloud['X'] >= x_start) & (cloud['X'] < x_start + tile_length) &
                    (cloud['Y'] >= y_start) & (cloud['Y'] < y_start + tile_length)]

                # ignore tiles, which are empty
                if tile.empty:
                    continue

                tile.to_csv(rf'E:\Grothum\debug\tile_{x_start}_{y_start}.txt', sep=',', index=False)

                iteration = 10
                weights = np.ones(tile.shape[0])

                for _ in range(iteration):
                    plane = fit_plane(tile, weights)
                    dist = calculate_distance_to_plane(plane, tile)
                    weights = weight_function(dist)

                # delete points, which are far off plane
                std = std_deviation(dist)
                selection = [1 if x < std * threshold_sigma else 0 for x in dist]
                tile = tile.iloc[selection]
                result_dfs.append(tile)
                # tile.to_csv(rf'E:\Grothum\debug\tile_{x_start}_{y_start}.txt', sep=',', index=False)

        # combine dfs
        df = pd.concat(result_dfs, ignore_index=True)
        # set df to dataframe
        if any(ele in df.columns for ele in ['R', 'G', 'B']):
            self.pointcloud = self._ndarray_to_cloud(df.values, file_structure=self.CloudFileStructure.XYZRGBS,
                                                     column_names=df.columns)
        else:
            self.pointcloud = self._ndarray_to_cloud(df.values, file_structure=self.CloudFileStructure.XYZS,
                                                     column_names=df.columns)

    def filter_cloud(self, NN_nbr: int, searchRadius: float, filterVariable: str, multiplierThreshStd: float = 1) -> \
    tuple[pd.DataFrame, float]:
        cloud = self.cloud_to_dataframe()
        columns = cloud.columns

        # building kdtree on xyz coordinated of pointcloud
        neighborsTree = NearestNeighbors(n_neighbors=NN_nbr, radius=searchRadius, algorithm='kd_tree').fit(
            cloud.iloc[:, :2])

        # find nearest neigbors of any point in pointcloud
        _, indices = neighborsTree.kneighbors(cloud.iloc[:, :2])
        colID = cloud.columns.get_loc(filterVariable)

        range_list = []
        for i in range(indices.shape[1]):
            range_list.append(cloud.iloc[indices[:, i], colID])
        nearest = np.asarray(range_list)
        nearestStd = pd.DataFrame(np.std(nearest.T, axis=1))

        threshVariable = np.nanmean(nearestStd) + multiplierThreshStd * np.nanstd(nearestStd)

        cloud['toFilter'] = nearestStd
        cloud = cloud[cloud['toFilter'] < threshVariable]
        del cloud['toFilter']

        self.pointcloud = self._ndarray_to_cloud(cloud.to_numpy(), file_structure=self.CloudFileStructure.XYZS,
                                                 column_names=columns)

        logging.info(rf'Filtered pointcloud {self.pointcloud.getName()} to {self.pointcloud.size()}'
                     rf'with NN_nbr: {NN_nbr}, Radius: {searchRadius}, filterVariable: {filterVariable}, '
                     rf'multiplierTrheshStd: {multiplierThreshStd},')

        return cloud, threshVariable

    def filter_local_cloud(self, NN_nbr: int, searchRadius: float, filterVariable: str,
                           multiplier_thresh_std: float = 1) -> pd.DataFrame:
        cloud = self.cloud_to_dataframe()
        columns = cloud.columns

        points_array = cloud.values
        # building kdtree on xyz coordinated of pointcloud
        neighborsTree = NearestNeighbors(n_neighbors=NN_nbr, radius=searchRadius, algorithm='kd_tree').fit(
            points_array[:, :2])

        # find nearest neigbors of any point in pointcloud
        # _, indices = neighborsTree.kneighbors(cloud.iloc[:, :2])
        # _, indices = neighborsTree.radius_neighbors(cloud.iloc[:,:2])
        colID = cloud.columns.get_loc(filterVariable)

        indices_to_keep = []

        for pts in points_array[:, :2]:
            iterate = True
            # _, indices = neighborsTree.radius_neighbors(pts.reshape(1,-1))
            _, indices = neighborsTree.kneighbors(pts.reshape(1, -1))
            indices = indices[0]

            if indices.shape[0] < 3:
                continue

            while iterate:
                neigb_points = points_array[indices, :]

                col_values_mean = neigb_points[:, colID].mean()
                col_values_std = neigb_points[:, colID].std()
                max_value_idx = np.argmax(neigb_points[:, colID])

                if neigb_points[max_value_idx, colID] > col_values_mean + col_values_std * multiplier_thresh_std:
                    indices = np.delete(indices, max_value_idx, axis=0)
                    # np.delete(neigb_points, indices[0][ max_value_idx],axis=0)
                else:
                    indices_to_keep.extend(list(indices))
                    break

        indices_to_keep = sorted(list(dict.fromkeys(indices_to_keep)))
        cloud = cloud.iloc[indices_to_keep, :]

        self.pointcloud = self._ndarray_to_cloud(cloud.to_numpy(), file_structure=self.CloudFileStructure.XYZS,
                                                 column_names=columns)

        logging.info(rf'Filtered pointcloud {self.pointcloud.getName()} to {self.pointcloud.size()}'
                     rf'with NN_nbr: {NN_nbr}, Radius: {searchRadius}, filterVariable: {filterVariable}, '
                     rf'multiplierTrheshStd: {multiplier_thresh_std},')

        return cloud

    def filter_cloud_color_vegetation(self, threshold):
        ptCloud = self.__cloud_to_dataframe()
        ptCloud['RdivG'] = ptCloud.R / ptCloud.G
        ptCloud = ptCloud[ptCloud['RdivG'] > threshold]
        del ptCloud['RdivG']
        self.pointcloud = self._ndarray_to_cloud(ptCloud.to_numpy(), E4DPointcloud.CloudFileStructure.XYZS,
                                                 column_names=ptCloud.columns)

    def filter_distance_to_plane(self, sigma_neg: float = 3.0, sigma_pos: float = 3.0,
                                 params=cloudComPy.Cloud2MeshDistancesComputationParams()):
        # fit plane through pointcloud
        plane = cc.ccPlane.Fit(self.pointcloud)
        # calculate_distance
        cc.DistanceComputationTools.computeCloud2MeshDistances(self.pointcloud, plane, params)

        df = self.cloud_to_dataframe()
        distanceCol = 'C2M absolute distances'
        std = df[distanceCol].std()
        df = df[(df[distanceCol] > (-1 * std * sigma_neg)) &
                (df[distanceCol] < (1 * std * sigma_pos))]

        df = df.drop(distanceCol, axis=1, errors='ignore')
        # set df to dataframe
        if all(ele in df.columns for ele in ['R', 'G', 'B', 'Alpha']):  # @TODO: write function
            self.pointcloud = self._ndarray_to_cloud(df.values, file_structure=self.CloudFileStructure.XYZRGBS,
                                                     column_names=df.columns)
        else:
            self.pointcloud = self._ndarray_to_cloud(df.to_numpy(), file_structure=self.CloudFileStructure.XYZS,
                                                     column_names=df.columns)

    def filter_noise(self, octree=None):

        refCloud = cc.CloudSamplingTools.noiseFilter(self.pointcloud,
                                                     kernelRadius=self.noise_filter_params.kernel_radius,
                                                     nSigma=self.noise_filter_params.sigma,
                                                     removeIsolatedPoints=self.noise_filter_params.remove_isolated_points,
                                                     useKnn=self.noise_filter_params.use_knn,
                                                     knn=self.noise_filter_params.knn,
                                                     useAbsoluteError=self.noise_filter_params.use_absolute_error,
                                                     absoluteError=self.noise_filter_params.absolute_error,
                                                     octree=octree)

        self.pointcloud = self.pointcloud.partialClone(refCloud)[0]

    def save_transformation(self, transform: cc.ccGLMatrix, filename: str):
        if transform:
            mat = np.array(transform.data()).reshape(4, 4)
        else:
            mat = np.array(self.transform_matrix.data()).reshape(4, 4)

        if not os.path.exists(os.path.dirname(filename)):
            os.mkdir(os.path.dirname(filename))

        np.savetxt(filename, mat, fmt='%.8f', delimiter=' ')

    def load_transformation(self, filename: str) -> cc.ccGLMatrix:
        mat = np.loadtxt(filename, delimiter=' ').ravel().tolist()
        return cc.ccGLMatrix(mat)

    def calculate_m3c2_pm(self, target_cloud: E4DPointcloud, core_cloud: E4DPointcloud = None,
                          scalar_field_index_ref_cloud: list[int] = [],
                          scalar_field_index_target_cloud: list[int] = [],
                          scale: list[float] = [1.000, 1.000, 1.000]) -> E4DPointcloud:

        if self.m3c2_processing_params_file is None:
            print("Set path to m3c2-parameter file first!")
            return

        if core_cloud is None:
            clouds = [self.pointcloud, target_cloud.pointcloud]
        else:
            clouds = [self.pointcloud, target_cloud.pointcloud, core_cloud.pointcloud]

        # check for len of sf - indexes
        if len(scalar_field_index_ref_cloud) != 3 or len(scalar_field_index_target_cloud) != 3:
            logging.info(rf'Scalarfield indexes length: {len(scalar_field_index_ref_cloud)} '
                         rf'and {len(scalar_field_index_target_cloud)}. Using M3C2 without Precision Maps.')
            result = cc.M3C2.computeM3C2(clouds, self.m3c2_processing_params_file)

        else:
            sfs = []
            # TODO: Check, if 3 sf each cloud is provided, cancel or ignore processing, if not
            for index in scalar_field_index_ref_cloud:
                sfs.append(self.pointcloud.getScalarField(index))
            for index in scalar_field_index_target_cloud:
                sfs.append(target_cloud.pointcloud.getScalarField(index))
            result = cc.M3C2.computeM3C2(clouds, self.m3c2_processing_params_file, sfs, scale)

        export_cloud = E4DPointcloud()
        export_cloud.pointcloud = result

        logging.info(rf'Finished M3C2 Processing for source: {self.pointcloud.getName()}, '
                     rf'target {target_cloud.pointcloud.getName()}, '
                     rf'core {core_cloud.pointcloud.getName()}')

        return export_cloud

    def calculate_m3c2_pm_cl(self, target_cloud: E4DPointcloud, core_cloud: str, out_cloud: str):
        path_source = self.path
        cloud_compare_path = r"C:\Program Files\CloudCompare\CloudCompare.exe"
        # M3C2_Command = (
        #     r'"C:\Program Files\CloudCompare\CloudCompare.exe" -SILENT -AUTO_SAVE OFF -C_EXPORT_FMT ASC -PREC 5 -o "'
        #     + file_1 + '" -o "' + file_2 + '" -M3C2 ' + M3C2_param
        #     + ' -REMOVE_NORMALS -SAVE_CLOUDS FILE "' + "'" + ' ' + "' '" + ' ' + "' '" + M3C2_file + "'" + '"')

        M3C2_Command = (rf'{cloud_compare_path} -AUTO_SAVE OFF -C_EXPORT_FMT ASC -PREC 5 -o {path_source} '
                        rf'-o {target_cloud.path} -M3C2 {self.m3c2_processing_params_file} -REMOVE_NORMALS '
                        rf'-SAVE_CLOUDS FILE "temp.txt temp2.txt {out_cloud}"')

        subprocess.run(M3C2_Command)

    def calculate_geotiff(self, gridstep: float, output_dir: str):

        cc.RasterizeGeoTiffOnly(self.pointcloud, gridStep=gridstep, outputRasterZ=True, outputRasterSFs=True,
                                pathToImages=output_dir)

    def _ndarray_to_polyline(self, array: np.ndarray):
        poly_cloud = self._ndarray_to_cloud(array)

        poly_line = cc.ccPolyline(poly_cloud)
        poly_line.addChild(poly_cloud)
        poly_line.addPointIndex(0, poly_cloud.size())
        poly_line.setClosed(True)

        return poly_cloud, poly_line

    def _ndarray_to_cloud(self, array: np.ndarray,
                          file_structure: CloudFileStructure = CloudFileStructure.XYZ,
                          column_names: list = []):

        assert array.shape[1] >= 3, 'Count of columns in ndarray must be 3 (xyz) or greater (with scalar fields)'

        # TODO: condensate code doubling into function (private memberfunction or lambda)
        if file_structure == E4DPointcloud.CloudFileStructure.XYZ:
            cloud = cc.ccPointCloud()
            cloud.reserve(array.shape[0])
            cloud.coordsFromNPArray_copy(array)

        elif file_structure == E4DPointcloud.CloudFileStructure.XYZS:
            cloud = cc.ccPointCloud()
            cloud.reserve(array.shape[0])
            cloud.coordsFromNPArray_copy(array[:, 0:3])
            for index, item in enumerate(column_names[3:], start=3):
                cloud.addScalarField(item)
                sf = cloud.getScalarField(item)
                sf.fromNpArrayCopy(array[:, index])

        elif file_structure == E4DPointcloud.CloudFileStructure.XYZRGBS:
            cloud = cc.ccPointCloud()
            cloud.reserve(array.shape[0])
            cloud.coordsFromNPArray_copy(array[:, 0:3])
            cloud.colorsFromNPArray_copy(array[:, 3:7])
            for index, item in enumerate(column_names[7:], start=3):
                cloud.addScalarField(item)
                sf = cloud.getScalarField(item)
                sf.fromNpArrayCopy(array[:, index])

        return cloud

    def cloud_to_ndarray(self) -> tuple[np.ndarray, list[str]]:
        # TODO: export normals into array, if exists
        array = self.pointcloud.toNpArrayCopy()
        column_names = ['X', 'Y', 'Z']

        if self.pointcloud.hasColors():
            array_color = self.pointcloud.colorsToNpArrayCopy()
            array = np.concatenate((array, array_color), axis=1)
            column_names.extend(['R', 'G', 'B', 'Alpha'])

        if self.pointcloud.hasScalarFields():
            sf_names = [item[0] for item in self.pointcloud.getScalarFieldDic().items()]
            column_names.extend(sf_names)
            array_sf = np.ndarray(shape=(array.shape[0], len(sf_names)))
            for index, name in enumerate(sf_names):
                sf = self.pointcloud.getScalarField(name).toNpArray()
                array_sf[:, index] = sf
            array = np.concatenate((array, array_sf), axis=1)

        return array, column_names

    def cloud_to_dataframe(self) -> pd.DataFrame:
        array, column_names = self.cloud_to_ndarray()
        return pd.DataFrame(array, columns=column_names)
