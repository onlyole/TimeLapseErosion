import math
import struct

import cv2
import exifread
from osgeo import gdal, osr
# import GDAL
# import gdal
import os
import numpy
import glob
import natsort
import re
import datetime
import time
import copy
import matplotlib.dates as mdates
import tkinter as tk
from tkinter import filedialog
# from mpl_toolkits import basemap
# import georaster
from PIL import Image
import pathlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib_scalebar.scalebar import ScaleBar
import pandas as pd
from tqdm import tqdm


def plotting_timeline(ax, data_collection, x_lim, y_names="average", time_var_name='captureTime', plot_name='plotChange',
                      error='stdDev', n=5, reset_time_file="", y_min=-0.1, y_max=0.1):
    # data_collection: collection of dataframes
    # plotting specs (i.e. linewidths, linestyles, markersizes, markertypes, labels) need to be listed

    # set font properties
    font_properties = {'size': 12, 'family': 'serif'}

    # set start and end time
    # start = pd.to_datetime(data_collection[0][time_var_name]).min()
    # end = pd.to_datetime(data_collection[0][time_var_name]).max()
    start = x_lim[0]
    end = x_lim[1]

    ax.set_xlim(left=start, right=end)

    for data in data_collection:
        # define axes for plotting
        x_axis = pd.to_datetime(data[time_var_name]).head(n=n)
        y_axis = data[y_names].head(n=n)

        # plot data
        ax.plot(x_axis, y_axis, '.', color='tab:brown')

        # if error:
        # if False:
        #     err = data[error].head(n=n)
        #     # ax.errorbar(x_axis, y_axis, yerr=err, fmt='-', alpha=.1)
        #     ax.errorbar(x_axis, y_axis, yerr=err, alpha=.1)

    # set reference reset
    if reset_time_file:
        resets = pd.read_csv(reset_time_file, sep=",", header=0)
        resets["date"] = pd.to_datetime(resets["date"])
        last_time = x_axis.max()

        for idx, row in resets.iterrows():
            if row["date"] < last_time:
                ax.vlines(row["date"], -0.1, 0.1)

    ax.set_ylim(y_min, y_max)

    # remove features of graph
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # set tick parameters
    ax.tick_params(direction='in', length=2, width=0.3, colors='k')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    # set axis width
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    # define dates for graph
    mdates.MonthLocator()  # every month
    mdates.DateFormatter('%m')
    days = mdates.DayLocator(interval=30)  # every day
    mdates.DateFormatter('%d')
    mdates.HourLocator()
    mdates.DateFormatter('%H')
    minutes = mdates.MinuteLocator(byminute=[0, 15, 30, 45])  # (interval=minute_interval)
    mdates.DateFormatter('%M')
    mdates.SecondLocator()
    mdates.DateFormatter('%S')

    # define date format, which will be put as tick label
    if end - start < pd.Timedelta(1, 'D'):
        ymd = mdates.DateFormatter('%H:%M:%S')
    elif end - start < pd.Timedelta(365, 'D'):
        ymd = mdates.DateFormatter('%Y-%m-%d')
    else:
        ymd = mdates.DateFormatter('%Y-%m-%d')

    # hms = mdates.DateFormatter('%H:%M:%S')

    # format the ticks
    # ax.xaxis.set_major_locator(minutes)
    # ax.xaxis.set_major_formatter(hms)
    ax.xaxis.set_major_formatter(ymd)
    # ax.xaxis.set_minor_locator(minutes)

    # define format of x-axis ticks label
    # ax.format_xdata = mdates.DateFormatter('%H:%M:%S')

    # write axis label
    ax.set_ylabel(' ', **font_properties)
    # ax.set_xlabel('date', **fontProperties)

    # fig.autofmt_xdate()

    # plt.show()
    # plt.savefig(os.path.join(output_dir, plot_name + '.pdf'), dpi=600)
    # plt.close(fig)


def copy_dataset(file, raster_band_nr, outFileName):
    ds = gdal.Open(file)
    band = ds.GetRasterBand(raster_band_nr)
    arr = band.ReadAsArray()
    [rows, cols] = arr.shape
    arr_min = arr.min()
    arr_max = arr.max()
    # arr_mean = int(arr.mean())
    # arr_out = numpy.where((arr < arr_mean), 10000, arr)
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outFileName, cols, rows, 1, gdal.GDT_UInt16)
    outdata.SetGeoTransform(ds.GetGeoTransform())  ##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())  ##sets same projection as input
    outdata = gdal.FillNodata(targetBand=outdata.GetRasterBand(raster_band_nr), maskBand=None,
                              maxSearchDist=50, smoothingIterations=20)  # interpolating nodata
    # outdata.GetRasterBand(1).WriteArray(arr_out)
    # outdata.GetRasterBand(1).SetNoDataValue(10000)  ##if you want these values transparent
    outdata.FlushCache()  ##saves to disk!!
    outdata = None
    band = None
    ds = None

    return outdata


def export_certain_band(input_dataset, output_path):
    """

    """
    fmttypes = {'Byte': 'B', 'UInt16': 'H', 'Int16': 'h', 'UInt32': 'I', 'Int32': 'i', 'Float32': 'f', 'Float64': 'd'}

    dataset = input_dataset
    prj = dataset.GetProjection()


    number_band = 2 # index of band, which contains m3c2 distances
    band = dataset.GetRasterBand(number_band)
    geotransform = dataset.GetGeoTransform()

    # Set name of output raster
    if number_band == 2:
        output_file = output_path

    # Create gtif file with rows and columns from parent raster
    driver = gdal.GetDriverByName("GTiff")

    columns, rows = (band.XSize, band.YSize)
    BandType = gdal.GetDataTypeName(band.DataType)

    raster = []

    for y in range(band.YSize):
        scanline = band.ReadRaster(0, y, band.XSize, 1, band.XSize, 1, band.DataType)
        values = struct.unpack(fmttypes[BandType] * band.XSize, scanline)
        raster.append(values)

    dst_ds = driver.Create(output_file,
                           columns,
                           rows,
                           1,
                           band.DataType)

    # flattened list of raster values
    raster = [item for element in raster for item in element]

    # transforming list in array
    raster = numpy.asarray(numpy.reshape(raster, (rows, columns)))

    ##writting output raster
    dst_ds.GetRasterBand(1).WriteArray(raster)

    # setting extension of output raster
    # top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
    dst_ds.SetGeoTransform(geotransform)

    # setting spatial reference of output raster
    srs = osr.SpatialReference(wkt=prj)
    dst_ds.SetProjection(srs.ExportToWkt())

    # Close output raster dataset
    dst_ds = None

    # Close main raster dataset
    dataset = None


def plot_stack_bands(input_vrt_stack, files_stack, avg_df, df_column_name='average', camera_image=None, fillEmpty=True, output_dir='.'):
    """
    Generates  a collage of m3c2 image, camera image and graph plot for any given m3c2 band in input_vrt_stack.
    """

    im_stack = []
    png_stack = []

    # times when a observation is accepted
    start_time = pd.to_datetime("09:00:00").time()
    end_time = pd.to_datetime("11:00:00").time()

    # filter dataframe for samples in between start and end time
    avg_df = avg_df.set_index("captureTime", drop=False).between_time('09:00:00', "11:00:00").reset_index(drop=True)
    count = 1

    for i in range(1, input_vrt_stack.RasterCount, 1):
        band = input_vrt_stack.GetRasterBand(i)
        img = camera_image[i-1]

        if img:
            #open camera image and export exif time
            with open(img, 'rb') as f:
                datetime_str = str(exifread.process_file(f)["EXIF DateTimeOriginal"])
                datetime_img = pd.to_datetime(datetime_str, format="%Y:%m:%d %H:%M:%S")
                year = datetime_img.year
                first_day_of_year = pd.Timestamp(year, 1, 1, 0)
                last_day_of_year = pd.Timestamp(year, 12, 31, 23, 59, 59)
                y_min = math.floor(avg_df[df_column_name].min() * 100) / 100
                y_max = math.ceil(avg_df[df_column_name].max() * 100) / 100

        # skip data, which is taken outside of daily window
        if not (start_time <= datetime_img.time() <= end_time):
            continue

        # fill empty cells by interpolation
        if fillEmpty:
            gdal.FillNodata(targetBand=band, maskBand=None,
                            maxSearchDist=5, smoothingIterations=0)

        # export m3c2 band as array
        arr_band = band.ReadAsArray()

        # prepare
        fig = plt.figure(layout="tight", figsize=(12,8))
        gs = plt.GridSpec(2, 2, figure=fig, height_ratios=[4, 1])
        ax_band = fig.add_subplot(gs[0, 0])
        ax_graph = fig.add_subplot(gs[1, :])
        ax_image = fig.add_subplot(gs[0, 1])

        ax_image.xaxis.set_tick_params(labelbottom=False)
        ax_image.yaxis.set_tick_params(labelleft=False)
        ax_image.set_xticks([])
        ax_image.set_yticks([])

        # fig, (ax1, ax2, ax3) = plt.subplots(2, 2, gridspec_kw={'height_ratios': [4, 1, 4]}, figsize=(12, 8), clear=False,
        #                                tight_layout=True)
        im1 = ax_band.imshow(arr_band, cmap='RdBu', animated=True, vmin=-0.1, vmax=0.1)

        # Hide X and Y axes label marks
        ax_band.xaxis.set_tick_params(labelbottom=False)
        ax_band.yaxis.set_tick_params(labelleft=False)

        # Hide X and Y axes tick marks
        ax_band.set_xticks([])
        ax_band.set_yticks([])

        # plot time on upper left
        time = ''

        # max y and min y from avg_df
        y_min = avg_df[df_column_name].min() + 0.1 * avg_df[df_column_name].min()

        for index, row in avg_df.iterrows():
            # print (f'{row["filename"]} in {files_stack[i-1]}')
            if row['filename'].split('.')[0] in files_stack[i - 1]:
                # print("found")
                time = row['captureTime']

        ax_band.text(0, -1, time, fontsize=12)

        divider = make_axes_locatable(ax_band)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')

        # im2 = ax2.plot([1,2,3], [1,2,3])
        plotting_timeline(ax_graph, [avg_df], x_lim=[first_day_of_year, last_day_of_year], n=count,
                          reset_time_file=r"D:\Grothum\Kamerstation_Metashape\reset_reference.txt",
                          y_min=y_min, y_max=0.01)

        # plot camera image
        if img:
            img_arr = cv2.imread(img, 1)
            ax_image.imshow(img_arr)

        fig.autofmt_xdate()
        plt.savefig(os.path.join(output_dir, f'{year}_{i}.png'), format='png')
        count += 1


    return im_stack


def interpolate_null_values(file_tiff, raster_band_nr):
    # nach https://gis.stackexchange.com/questions/151020/how-to-use-gdal-fillnodata-in-python

    ET = gdal.Open(file_tiff, gdal.GA_Update)
    ETband = ET.GetRasterBand(raster_band_nr)

    result = gdal.FillNodata(targetBand=ETband, maskBand=None,
                             maxSearchDist=10, smoothingIterations=2)

    ET = None
    return result


def collect_images_and_time(image_root_path, camera_name):
    images = glob.glob(os.path.join(image_root_path, '*', camera_name, '*.jpg'))

    data = []
    for image in images:
        f = open(image, 'rb')
        tags = exifread.process_file(f)
        datetime = tags["EXIF DateTimeOriginal"]
        data.append([image, datetime])

    df = pd.DataFrame(data)
    df.columns = ["image_path", "datetime"]
    df["datetime"] = pd.to_datetime(df["datetime"])

    return images


def collect_time_attribution_files(root_path):
    files = glob.glob(os.path.join(root_path, 'Time_Attribution_*.txt'))

    dataframes = []
    for file in files:
        dir_path = os.path.dirname(file)

        def change_path_to(source_path, target_path):
            try:
                src = pathlib.Path(source_path)
                tgt = pathlib.Path(target_path)

                last_tgt = tgt.parts[-1]
                to_img = src.parts[src.parts.index(last_tgt) + 1:]

                return str(tgt.joinpath(*to_img))
            except:
                return source_path


        df = pd.read_csv(file, header=0, sep=',')
        date_cols = [c for c in df.columns if 'datetime' in c]
        path_cols = [c for c in df.columns if 'path' in c]
        df[date_cols] = df[date_cols].apply(pd.to_datetime, unit='s', origin='unix')
        df[path_cols] = df[path_cols].map(change_path_to, target_path=dir_path)
        min_t, max_t = df[date_cols[0]].min(), df[date_cols[0]].max()
        dataframes.append([df, min_t, max_t])

    return dataframes


def find_image_in_time_attribution(dataframes, time):
    path = None
    for dataframe in dataframes:
        found = False
        if pd.isna(dataframe[1]) or pd.isna(dataframe[2]):
            continue

        if dataframe[1] <= time <= dataframe[2]:
            found = True
            ...  # find closest time
            df = dataframe[0]
            time_col = [c for c in df.columns if "datetime" in c][0]
            idx = df[time_col].searchsorted(time)
            path = df.iloc[idx, 3]  # 3 is column 4: path to image
            break

    return path


if __name__ == '__main__':

    # Configuration
    root_path = r"D:\Grothum\Kamerstation_Metashape\Station_unten\m3c2"
    image_path = r"D:\Grothum\Kamerastationen\Station_unten"
    dfs = collect_time_attribution_files(image_path)
    geotiff_path = os.path.join(root_path, "geotiff")
    timeseries_file = os.path.join(root_path, "timeseries", "statsChange.txt")
    update_single_band_tiffs = False

    geotiff_files = glob.glob(os.path.join(geotiff_path, "*.tif"))
    geotiff_files = [file for file in geotiff_files if
                     not any(substr in file for substr in ["singleBand", "distances"])]
    geotiff_files = natsort.natsorted(geotiff_files,
                                      key=lambda x: re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}", x)[1])

    m3c2_avg_df = pd.read_csv(timeseries_file, header=0, sep=",")
    m3c2_avg_df['captureTime'] = pd.to_datetime(m3c2_avg_df['captureTime'], format='%Y-%m-%d %H:%M:%S')
    col_name = 'average'

    timeseries_arr = []
    ims = []
    files_distances = []
    files_single_band = []
    files_camera = []
    # iterate over files
    for filepath in tqdm(geotiff_files, unit="file"):
        try:
            # create editable copy
            fileOut = os.path.abspath(filepath).split('.')[0] + "_distances.tif"
            file_single_band = os.path.abspath(filepath).split('.')[0] + "_singleBand.tif"

            # extract time from file
            current_time = pd.to_datetime(re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}", filepath)[1],
                                          format="%Y-%m-%dT%H-%M-%S")
            files_camera.append(find_image_in_time_attribution(dfs, current_time))

            if os.path.exists(fileOut) and not update_single_band_tiffs:
                files_distances.append(fileOut)

            else:
                file_format = "GTiff"
                driver = gdal.GetDriverByName(file_format)

                src_ds = gdal.Open(filepath)
                dst_ds = driver.CreateCopy(fileOut, src_ds, strict=0)

                # band = dst_ds.GetRasterBand(4)
                # result = gdal.FillNodata(targetBand=band, maskBand=None,
                #                       maxSearchDist=5, smoothingIterations=0)

                src_ds = None
                dst_ds = None

                # interpolate on no values
                interpolate_null_values(fileOut, 2)

                # export
                files_distances.append(fileOut)

        except Exception as e:
            print(e)
            continue

        if os.path.exists(file_single_band) and not update_single_band_tiffs:
            files_single_band.append(file_single_band)
            raster = None

        else:
            raster = gdal.Open(fileOut, gdal.GA_ReadOnly)
            export_certain_band(raster, file_single_band)
            files_single_band.append(file_single_band)

            raster = None

    # stack those dgm together
    root_path = os.path.dirname(geotiff_files[0])
    vrt = "output_stack.vrt"
    gdal.BuildVRT(vrt, files_single_band, separate=True, bandList=[4], callback=gdal.TermProgress_nocb)

    stack = gdal.Open(vrt, gdal.GA_Update)

    # generate output dir with actual time
    now = datetime.datetime.now()
    output_dir = os.path.join(root_path, f'{now.strftime("%Y-%m-%dT%H-%M-%S")}_{col_name}')
    os.makedirs(output_dir, exist_ok=True)
    im_stack = plot_stack_bands(stack, files_single_band, m3c2_avg_df, camera_image=files_camera, df_column_name=col_name,
                                output_dir=output_dir)
    fig_ani = plt.figure()

    print(im_stack)
    print(type(im_stack))

    time.sleep(10)
    # delete temp files
    vrt = None
    stack = None
