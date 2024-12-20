import os
import re
import glob
import pandas as pd
import numpy as np

"""
Script to create an overview of all processed dense pointclouds, precision maps and filtered pointclouds 
per station and month
"""

if __name__ == '__main__':

    root_dir = r'D:\Grothum\Kamerstation_Metashape'

    stations = ['Station_oben', 'Station_mitte', 'Station_unten']

    station_stats = {}
    for st in stations:
        station_dir = os.path.join(root_dir, st)
        months = [mth for mth in glob.glob(os.path.join(station_dir, '*')) if
                bool(re.search('\d{4}-\d{2}', mth)) and os.path.isdir(mth)]

        mth_counts = []
        for mth in months:
            # count densed, filtered and prec maps
            densed = [file for file in glob.glob(os.path.join(root_dir, st, mth, 'Dense', '*.*'))]
            filtered = [file for file in glob.glob(os.path.join(root_dir, st, mth, 'Dense', 'filtered', '*.*'))]
            prec = [file for file in glob.glob(os.path.join(root_dir, st, mth, 'ptPrecision', '*.*'))]
            mth_counts.append([os.path.basename(mth), len(densed), len(prec), len(filtered)])

        station_stats[st] = mth_counts

    # create dictionaries and write to excel
    excel_file_path = os.path.join(root_dir, "overview.xlsx")
    with pd.ExcelWriter(excel_file_path) as writer:
        for st in stations:
            df = pd.DataFrame(station_stats[st], columns=['month', 'dense_count', 'precision_count', 'filtered_count'])
            df.to_excel(writer, sheet_name=st, index=False)