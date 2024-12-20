import os
from definitions import DEMO_DIR
import pandas as pd
import glob
import matplotlib.pyplot as plt

# from fastdtw import fastdtw
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error

from src.core.E4DImage import E4DPicture
# Configuration
station = "Station_oben"  # Setting of experiment_name
# root_path = rf'D:\Grothum\e4d_demo\Kamerastationen\{station}'  # root_path to images, images under {root_path}/YYYY-MM/{cam_name}/
root_path = rf'{DEMO_DIR}\e4d_demo\Kamerastationen\{station}'  # root_path to images, images under {root_path}/YYYY-MM/{cam_name}/
months = ["2020-06", "2020-07", "2020-08"]  # months to process

# setting camera names according to experiment name.
station_cams = {'Station_oben': ["Kamera21", "Kamera22", "Kamera23", "Kamera24", "Kamera25"],
                'Station_mitte': ["Kamera11", "Kamera12", "Kamera13", "Kamera14", "Kamera15"],
                'Station_unten': ["Kamera1", "Kamera2", "Kamera3", "Kamera4", "Kamera5"]}

cams = station_cams[station]


def process_camera_images(root_path, camera, months):
    """
    Processes images for a single camera, loading their metadata into a DataFrame.
    """
    data_cam_set = []
    for month in months:
        image_files = glob.glob(os.path.join(root_path, month, camera, '*.jpg'))
        for item in image_files:
            with E4DPicture(item) as img:
                data_cam_set.append([os.path.basename(item), img.get_img_datetime(), item])
    df = pd.DataFrame(data_cam_set, columns=['pic', 'datetime', 'path'])
    df.Name = os.path.basename(camera)
    return df


def remove_empty_cameras(cams, data_frames):
    """
    Removes cameras that have no images from the camera list.
    """
    cams_to_remove = [camera for camera, df in zip(cams, data_frames) if df.empty]
    for camera in cams_to_remove:
        cams.remove(camera)
    return cams_to_remove


def reformat_datetime(data_frames):
    """
    Reformats the 'datetime' column of each DataFrame.
    """
    for df in data_frames:
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y:%m:%d %H:%M:%S')


def get_reference_frame(data_frames):
    """
    Identifies the reference DataFrame with the most image entries.
    """
    entry_count = [df.shape[0] for df in data_frames]
    idx_max = entry_count.index(max(entry_count))
    ref = data_frames.pop(idx_max)
    return ref, data_frames


def filter_outliers(x, y, y_pred, rmse):
    x_temp = []
    y_temp = []
    for idx, (x, y, y_pred) in enumerate(zip(x, y, y_pred)):
        if abs(y - y_pred) < (rmse * 3):
            x_temp.append(x)
            y_temp.append(y)
        else:
            print(f'Remove item pair {x}, {y}')
    return x_temp, y_temp



def append_border_dates(ref_epoch, x, line_params, y_pred, t_start, t_end):
    if not (t_start.day == pd.Timestamp(ref_epoch + pd.to_timedelta(x[0], unit='sec')).day) & (
            t_start.month == pd.Timestamp(ref_epoch + pd.to_timedelta(x[0], unit='sec')).month):
        print("put beginning in front")
        x = np.insert(x, 0, (t_start - ref_epoch).total_seconds(), axis=0)
        y_pred = np.insert(y_pred, 0, line_params.intercept + line_params.slope * (t_start - ref_epoch).total_seconds(),
                           axis=0)

    if not (t_end.day == pd.Timestamp(ref_epoch + pd.to_timedelta(x[-1], unit='sec')).day) & (
            t_end.month == pd.Timestamp(ref_epoch + pd.to_timedelta(x[-1], unit='sec')).month):
        print("put end in back")
        x = np.insert(x, x.shape[0], (t_end - ref_epoch).total_seconds(), axis=0)
        y_pred = np.insert(y_pred, y_pred.shape[0],
                           line_params.intercept + line_params.slope * (t_end - ref_epoch).total_seconds(), axis=0)

    return x, y_pred


# collect images and times in dataframes for each camera
data_frames = [process_camera_images(root_path, camera, months) for camera in cams]
remove_empty_cameras(cams, data_frames)
reformat_datetime(data_frames)
ref, sub_frames = get_reference_frame(data_frames)

# group reference frame into groups on daily base
ref_groups = ref.groupby(pd.Grouper(key='datetime', freq='D'))
ref_epoch = pd.Timestamp(1970, 1, 1, 0)
attribution_table = ref.copy(deep=True)
attribution_table = attribution_table.rename(
    {'pic': f'{ref.Name}_pic', 'datetime': f'{ref.Name}_datetime', 'path': f'{ref.Name}_path'}, axis='columns')

# create table of all images to export as backup
columns = attribution_table.columns.to_list()
for sub in sub_frames:
    columns.extend([f'{sub.Name}_pic', f'{sub.Name}_datetime', f'{sub.Name}_path'])
for mth in months:
    mth_sub_frames = []
    for sub in sub_frames:
        mth_sub_frames.append(sub[(sub['datetime'].dt.month == pd.to_datetime(mth).month)])

    backup_df = pd.concat([ref, *mth_sub_frames], axis=1, ignore_index=True)
    backup_df.columns = columns
    backup_df.to_csv(os.path.join(root_path, f'ImageList_{mth}.txt'), index=False, sep=',')

revoke_subframes = []

for idx, sub in enumerate(sub_frames):
    print(f'Aligning pictures for cam {sub.Name}')
    zero_times = []
    one_times = []

    time_first = None
    time_last = None

    # group sub frame into groups on daily base
    sub_groups = sub.groupby(pd.Grouper(key='datetime', freq='D'))
    for date, sub_group in sub_groups:
        try:
            ref_group = ref_groups.get_group(date)
        except:
            continue

        # locate pictures taken between 9:00 and 11:00 a.m.
        sub_index = pd.DatetimeIndex(sub_group['datetime'])
        sub_group = sub_group.iloc[sub_index.indexer_between_time('9:30', '11:00')]
        ref_index = pd.DatetimeIndex(ref_group['datetime'])
        ref_group = ref_group.iloc[ref_index.indexer_between_time('9:30', '11:00')]

        if sub_group.shape[0] < 1:
            continue

        if sub_group["datetime"].count() <= 2 and ref_group["datetime"].count() <= 2:
            sub_group = sub_group.reset_index()
            ref_group = ref_group.reset_index()

            dt = (ref_group["datetime"] - sub_group["datetime"]).dt.total_seconds()
            if not np.isnan(dt.iloc[0]):
                zero_times.append(((date - ref_epoch).total_seconds(), dt.iloc[0]))

            if dt.count() == 2:
                if not np.isnan(dt.iloc[1]):
                    one_times.append(((date - ref_epoch).total_seconds(), dt.iloc[1]))

            # track date range of dataset
            if not time_first:
                time_first = date
                time_last = date
            else:
                time_last = date

    # fit line through each series and detect outliers

    x_zero = np.array([x[0] for x in zero_times])
    y_zero = np.array([x[1] for x in zero_times])
    x_one = np.array([x[0] for x in one_times])
    y_one = np.array([x[1] for x in one_times])

    rmse_zero = 0
    rmse_one = float("inf")

    # outlier detection iterative
    iterations = 5

    print(f"length of zero series: {x_zero.shape[0]}, {y_zero.shape[0]}")
    print(f"length of one series: {x_one.shape[0]}, {y_one.shape[0]}")
    print(y_zero)
    print(y_one)

    if x_zero.shape[0] < 3:
        revoke_subframes.append(sub.Name)
        continue

    while iterations:

        # use model for prediction on times and calculate rmse through
        result_zero = stats.linregress(x_zero, y_zero)
        y_pred_zero = result_zero.intercept + result_zero.slope * x_zero
        rmse_zero = mean_squared_error(y_true=y_zero, y_pred=y_pred_zero, squared=False)
        print('zero RMSE: {:.3f}'.format(rmse_zero))
        # iterate through points and remove outlieres
        x_zero_temp, y_zero_temp = filter_outliers(x_zero, y_zero, y_pred_zero, rmse_zero)
        x_zero = np.array(x_zero_temp)
        y_zero = np.array(y_zero_temp)
        y_pred_zero = result_zero.intercept + result_zero.slope * x_zero

        if x_one.size > 2:
            result_one = stats.linregress(x_one, y_one)
            y_pred_one = result_one.intercept + result_one.slope * x_one
            rmse_one = mean_squared_error(y_true=y_one, y_pred=y_pred_one, squared=False)
            print('one RMSE: {:.3f}'.format(rmse_one))
            x_one_temp, y_one_temp = filter_outliers(x_one, y_one, y_pred_one, rmse_one)
            x_one = np.array(x_one_temp)
            y_one = np.array(y_one_temp)
            y_pred_one = result_one.intercept + result_one.slope * x_one

        iterations -= 1

    plt.figure()
    plt.plot(x_zero, y_zero, 'o')
    plt.plot(x_one, y_one, 'o')
    if rmse_zero < rmse_one:
        plt.plot(x_zero, y_pred_zero)
    else:
        plt.plot(x_one, y_pred_one)
    plt.title(f"{sub.Name} to {ref.Name} Time Attribution")
    plt.savefig(os.path.join(root_path,
                             f'{ref.Name}_to_{sub.Name}_{time_first.strftime("%Y-%m")}_{time_last.strftime("%Y-%m")}_fit.png'))

    # append first day and last day of month to arrays if neccessary
    # create datetime index for whole month
    t_start = pd.to_datetime(months[0], format='%Y-%m')
    p_last_month = pd.Period(months[-1], freq='D')
    t_end = pd.to_datetime(months[-1], format='%Y-%m').replace(day=p_last_month.days_in_month)

    # check, if all days are represent and expand first and last day if necessary
    print(f't_start: {t_start} vs {time_first}')
    print(f't_end: {t_end} vs {time_last}')

    x_zero, y_pred_zero = append_border_dates(ref_epoch, x_zero, result_zero, y_pred_zero, t_start, t_end)
    if x_one.size > 2:
        x_one, y_pred_one = append_border_dates(ref_epoch, x_one, result_one, y_pred_one, t_start, t_end)

    # concat x and y_pred together, append with first/last day of month, if not included
    if (rmse_zero < rmse_one) or (len(x_one) < 3):
        print("use zero series")
        rmse = rmse_zero
        arr = np.concatenate((x_zero, y_pred_zero)).reshape((-1, 2), order='F')
    else:
        print("use one series")
        rmse = rmse_one
        arr = np.concatenate((x_one, y_pred_one)).reshape((-1, 2), order='F')

    print(pd.Timestamp(1970, 1, 1, 0) + pd.to_timedelta(arr[0, 0], unit='sec'),
          pd.Timestamp(1970, 1, 1, 0) + pd.to_timedelta(arr[-1, 0], unit='sec'))
    # map offset value for each day in month
    offset_frame = pd.DataFrame(arr, columns=['datetime', 'offset'])
    offset_frame['datetime'] = pd.Timestamp(1970, 1, 1, 0) + pd.to_timedelta(offset_frame['datetime'], unit='sec')
    offset_frame['rmse'] = rmse

    offset_frame = offset_frame.set_index('datetime')
    # interpolate on daily series
    offset_frame = offset_frame.resample('D').interpolate()
    print(offset_frame.head())
    print(offset_frame.tail())
    # iterate though ref pictures and link every subsequent picture to ref, if possible
    linked_list = []

    for idx, row in ref.iterrows():
        # gather day and offset
        day = row['datetime']
        offset = offset_frame.loc[day.normalize()]['offset']
        rmse = offset_frame.loc[day.normalize()]['rmse']

        # select sub rows based on actual day and iterate over day to find best fitting picture
        sub_query = sub.loc[sub['datetime'].dt.normalize() == day.normalize()]
        linked_img = None
        for _, subrow in sub_query.iterrows():
            if abs((subrow['datetime'] - day).total_seconds() + offset) < 3 * rmse:
                linked_img = [subrow['pic'], (subrow['datetime'] - ref_epoch).total_seconds(), subrow['path']]
                break
        # if no suitable image has been found, fill row with empty strings
        if linked_img is None:
            linked_img = ('', '', '')

        linked_list.append(linked_img)

    sub_df = pd.DataFrame(linked_list, columns=[f'{sub.Name}_pic', f'{sub.Name}_datetime', f'{sub.Name}_path'])
    attribution_table = pd.concat([attribution_table, sub_df], axis=1)

for camera in revoke_subframes:
    cams.remove(camera)

# group by month
month_groups = attribution_table.groupby(pd.Grouper(key=f'{ref.Name}_datetime', freq='M'))

for group in month_groups:
    # copy dataframe and convert reference datetime to utc seconds
    mth = group[0].strftime("%Y-%m")
    df = group[1].copy(deep=True)
    df = df.reset_index(drop=True)
    df[f'{ref.Name}_datetime'] = (df[f'{ref.Name}_datetime'] - ref_epoch).dt.total_seconds()
    df[f'{ref.Name}_datetime'] = df[f'{ref.Name}_datetime'].apply(lambda x: '{:.1f}'.format(x))
    # filter rows with a lot of nans
    # filter entries, which only have less then 3 pictures taken
    threshold = 6  # only to images with 3 columns of entries
    count_empty_strings = lambda row: sum(1 for value in row if value == '')
    df['empty_string_count'] = df.apply(count_empty_strings, axis=1)
    filtered_df = df[df['empty_string_count'] <= threshold]
    filtered_df = filtered_df.drop('empty_string_count', axis=1)
    filtered_df.reset_index(drop=True, inplace=True)
    filtered_df.to_csv(os.path.join(root_path, f"Time_Attribution_{station}_{mth}.txt"), index=True)


attribution_table[f'{ref.Name}_datetime'] = (attribution_table[f'{ref.Name}_datetime'] - ref_epoch).dt.total_seconds()
attribution_table[f'{ref.Name}_datetime'] = attribution_table[f'{ref.Name}_datetime'].apply(
    lambda x: '{:.1f}'.format(x))

threshold = 6  # only to images with 3 columns of entries
count_empty_strings = lambda row: sum(1 for value in row if value == '')
attribution_table['empty_string_count'] = attribution_table.apply(count_empty_strings, axis=1)
attribution_table = attribution_table[attribution_table['empty_string_count'] <= threshold]
attribution_table = attribution_table.drop('empty_string_count', axis=1)
attribution_table.reset_index(drop=True, inplace=True)

# %%
# plot cameratime on chunklabel and cameracount for each chunk
plt.style.use('seaborn')
count = attribution_table[[f'{x}_datetime' for x in cams]].apply(pd.to_numeric).count(axis=1)
table = attribution_table[[f'{x}_datetime' for x in cams]].apply(pd.to_numeric)
table = table.apply(lambda x: pd.to_timedelta(x, unit='s') + ref_epoch)
print(table.head())
# attribution_table[["Kamera21_datetime", "Kamera22_datetime", "Kamera23_datetime", "Kamera24_datetime", "Kamera25_datetime"]].head()
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
count.plot(ax=ax2, style=['b.'])
table.plot(ax=ax1, style=['r*'])

ax1.set_title(f'Time Attribution for {station} on {time_first.strftime("%Y-%m")}_{time_last.strftime("%Y-%m")}')
ax1.set_ylabel('Trigger Time Cameras')
ax1.xaxis.set_tick_params(labelbottom=False)
ax2.set_xlabel('Chunk')
ax2.set_ylabel('# Photos')
ax2.set_ylim(0, 6)
fig.savefig(os.path.join(root_path, f'Time_Attribution_{station}_{time_first.strftime("%Y")}'))
