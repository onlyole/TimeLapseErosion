import pandas as pd
import os, fnmatch
import re
import natsort
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.nonparametric.smoothers_lowess
from src.core.E4DPointcloud import E4DPointcloud
from tqdm import tqdm


def ploting_timeline(data_collection, y_names, output_dir, time_var_name, plot_name='plot', error=None):
    # data_collection: collection of dataframes
    # plotting specs (i.e. linewidths, linestyles, markersizes, markertypes, labels) need to be listed

    # set font properties
    fontProperties = {'size': 12,
                      'family': 'serif'}
    matplotlib.rc('font', **fontProperties)

    # prepare figure

    for idx, data in enumerate(data_collection):
        start = pd.to_datetime(data[time_var_name]).min()
        end = pd.to_datetime(data[time_var_name]).max()
        fig, ax = plt.subplots(1, 1)
        ax.set_xlim(left=start, right=end)
        # define axes for plotting
        x_axis = pd.to_datetime(data[time_var_name])
        y_axis = data[y_names]

        # plot data
        ax.plot(x_axis, y_axis, '.', color='tab:brown')

        # if error:
        # if False:
        #     err = data[error].head(n=n)
        #     # ax.errorbar(x_axis, y_axis, yerr=err, fmt='-', alpha=.1)
        #     ax.errorbar(x_axis, y_axis, yerr=err, alpha=.1)

        # set reference reset

        ax.set_ylim(-0.2, 0.05)

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
            ymd = mdates.DateFormatter('%m-%d %H:%M:%S')

        # hms = mdates.DateFormatter('%H:%M:%S')

        # format the ticks
        # ax.xaxis.set_major_locator(minutes)
        # ax.xaxis.set_major_formatter(hms)
        ax.xaxis.set_major_formatter(ymd)
        # ax.xaxis.set_minor_locator(minutes)

        # define format of x-axis ticks label
        # ax.format_xdata = mdates.DateFormatter('%H:%M:%S')

        # write axis label
        ax.set_ylabel(' ', **fontProperties)
        # ax.set_xlabel('date', **fontProperties)

        # fig.autofmt_xdate()

        # plt.show()
        plt.savefig(os.path.join(output_dir, plot_name + str(idx) + '.png'), dpi=600)
        # plt.close(fig)


def files_in_dir(dir_files, file_extension=None):
    files = os.listdir(dir_files)
    file_list = []
    for file in files:
        if fnmatch.fnmatch(file, '*' + file_extension):
            file_split = file.split('.')
            file_split = file_split[0].split('_')

            # fileSplit = fileSplit[3].split('-')
            file_split = file_split[1].split('-')

            file_list.append([int(file_split[-1]), file])
    file_list_sorted = sorted(file_list, key=lambda x: x[0])
    file_list_sorted_df = pd.DataFrame(file_list_sorted, columns=['count', 'name'])
    file_list_sorted_df = file_list_sorted_df.set_index('count')
    return file_list_sorted_df


if __name__ == '__main__':

    '''set parameters'''

    directory_m3c2 = r'D:\Grothum\Kamerstation_Metashape\Station_unten\m3c2'
    crop_dir = os.path.join(directory_m3c2, 'crop')
    directory_output = os.path.join(directory_m3c2, 'timeseries')
    if not os.path.exists(directory_output):
        os.mkdir(directory_output)

    # cameras to get time
    m3c2FileList = [(file, re.findall("\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}", file)[1]) for file in
                    glob.glob(os.path.join(directory_m3c2, '*.txt'))]
    m3c2FileList = natsort.natsorted(m3c2FileList, key=lambda x: x[1])
    m3c2FileList = [x[0] for x in m3c2FileList]
    # sort file list by second date

    plotOnly = False

    # variable to draw
    plotVar = 'average'
    plotVarDev = 'stdDev'

    # filter timeline
    quantileFilter = 0.002
    fracLOWESS = 0.02

    # draw only specific duration
    cutDate = False
    if cutDate:
        cutDateMin = '2022-07-14 11:30:00'
        cutDateMax = '2021-07-14 14:20:00'

    '''start processing'''
    statsChange = []
    output_file = os.path.join(directory_output, 'statsChange.txt')


    for file_m3c2 in tqdm(m3c2FileList):

        cloud = E4DPointcloud()
        cloud.load_pointcloud(file_m3c2, E4DPointcloud.CloudFileStructure.XYZS)

        for file in glob.glob(os.path.join(crop_dir, '*.txt')):
            cloud.crop_pointcloud(dim=2, points_filename=file, inside=True)

        # read m3c2 file to get stats of change
        # m3c2 = pd.read_csv(file_m3c2, header=0, index_col=False, sep=',')
        m3c2 = cloud.cloud_to_dataframe()
        # m3c2.columns = ['x', 'y', 'z', 'signChange', 'distUncertain', 'distM3C2', 'None']
        # m3c2.columns = ['X', 'Y', 'Z', 'sign', 'sigma', 'm3c2_ori', 'm3c2_mean', 'm3c2_median', 'm3c2_Nv', 'm3c2_time_weight']
        m3c2 = m3c2.rename(columns={"M3C2 distance": "distM3C2", "significant change": "signChange"})

        # remove lower and upper values, which usually outlier due to vegetation
        m3c2Quantile = m3c2.distM3C2.quantile(q=[quantileFilter, 1 - quantileFilter])
        m3c2 = m3c2[m3c2.distM3C2 > m3c2Quantile.iloc[0]]
        m3c2 = m3c2[m3c2.distM3C2 < m3c2Quantile.iloc[1]]

        # calculate stats
        averageM3C2 = m3c2.distM3C2.mean()
        stdM3C2 = m3c2.distM3C2.std()
        averageM3C2Sign = m3c2[m3c2.signChange == 1].distM3C2.mean()
        stdM3C2Sign = m3c2[m3c2.signChange == 1].distM3C2.std()

        # date extraction from filename
        date = re.findall("\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}", file_m3c2)[1]
        date = pd.to_datetime(date, format='%Y-%m-%dT%H-%M-%S')

        # keep date and stats
        statsChange.append(
            [os.path.basename(file_m3c2), date, averageM3C2, stdM3C2, averageM3C2Sign,
             stdM3C2Sign])

        # if count % 10 == 0:
        #     print('read ' + str(count) + ' files of ' + str(len(m3c2FileList)) + ' total files')
        #     print(processFile[cameraToGetTimeFrom])
        # count = count + 1

    statsChange = pd.DataFrame(statsChange)
    statsChange.columns = ['filename', 'captureTime', 'average', 'stdDev', 'averageSign', 'stdDevSign']

    statsChange.to_csv(output_file, index=False)

    statsChange = pd.read_csv(output_file, index_col=None)
    statsChange["captureTime"] = pd.to_datetime(statsChange["captureTime"], format="%Y-%m-%d %H:%M:%S")

    # remove lower and upper values, which potential outlier due to alignment issues
    statsChangeQuantile = statsChange[plotVar].quantile(q=[0.005, 1 - 0.005])
    statsChange = statsChange[statsChange[plotVar] > statsChangeQuantile.iloc[0]]
    statsChange = statsChange[statsChange[plotVar] < statsChangeQuantile.iloc[1]]
    statsChange[plotVar] = statsmodels.nonparametric.smoothers_lowess.lowess(statsChange[plotVar],
                                                                             statsChange.index.values,
                                                                             is_sorted=True, frac=fracLOWESS,
                                                                             return_sorted=False)

    # if cutDate:
    #     statsChange = statsChange[(statsChange.captureTime < cutDateMax)]
    #     statsChange = statsChange[(statsChange.captureTime > cutDateMin)]

    ploting_timeline([x[1] for x in statsChange.groupby(statsChange["captureTime"].dt.year)], plotVar,
                     directory_output, 'captureTime', 'plotChange', plotVarDev)
