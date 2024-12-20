import configparser
import sys
from src.core.CloudPair import CloudPair
from src.core.E4DPointcloud import E4DPointcloud
import definitions
import cloudComPy as cc

class PointCloudProcessing:

    def __init__(self, cl_pair: CloudPair):
        self.pc: E4DPointcloud = E4DPointcloud()
        self.pm_cloud: E4DPointcloud = E4DPointcloud()

        # load dense cloud
        self.pc.load_pointcloud(cl_pair.dense)
        # load sparse cloud (precision maps)
        self.pm_cloud.load_pointcloud(cl_pair.prec, file_structure=E4DPointcloud.CloudFileStructure.XYZS,
                                      delimiter='\t')

        self.crop_files = []

    def filter_pointcloud_on_scalarfield(self, N_nbnr, r, filter_variable, multiplier_thresh_std):
        """"""
        cloud = self.pm_cloud.filter_local_cloud(NN_nbr=N_nbnr, searchRadius=r, filterVariable=filter_variable,
                                                 multiplier_thresh_std=multiplier_thresh_std)

        return self

    def crop_dense(self):
        for file in self.crop_files:
            inside = True if 'aoi' in file else False
            self.pc.crop_pointcloud(dim=2, points_filename=file, inside=inside)

    # rasterize cloud
    def raster_dense(self, grid_step: float):
        self.pc.raster_pointcloud(grid_step=grid_step, vert_dir=cc.CC_DIRECTION.Z)

    # interpolate precision maps onto dense cloud
    def interpolate_prec_on_dense(self, radius, sigma):
        interpolate_params = cc.interpolatorParameters()
        interpolate_params.method = cc.INTERPOL_METHOD.RADIUS
        interpolate_params.algos = cc.INTERPOL_ALGO.NORMAL_DIST
        interpolate_params.radius = 1.0
        interpolate_params.sigma = 0.3
        self.pc.interpolate_scalar(self.pm_cloud.pointcloud,
                                   range(0, self.pm_cloud.pointcloud.getNumberOfScalarFields()),
                                   interpolate_params)


def create_m3c2_config_file(root_config_path: str, log_path: str, export_path: str):
    # read log file and return cp 3d error as measurement of registration error
    with open(log_path) as file:
        for line in file:
            if 'Marker point residual error 3D (CP)' in line:
                err = float(line.split(':')[1])
                break

    # read root m3c2 config, set registration error
    root_config = configparser.ConfigParser()
    root_config.read(root_config_path)
    root_config.set('General', 'RegistrationError', str(err))
    root_config.set('General', 'RegistrationErrorEnabled', 'true')
    # save to cloud specific m3c2 parameter file
    with open(export_path, 'w') as configfile:
        root_config.write(configfile)
