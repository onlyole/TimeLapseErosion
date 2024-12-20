import PIL.Image
import PIL.ExifTags
import os, glob
import pandas as pd
from collections import Counter
import pathlib
import tqdm
import shutil
import re


def check_word_in_path(path: str, word: str) -> bool:
    """Checks, if a complete word is a subdirectory in the path. Substrings are neglected"""
    # Define the regex pattern with word boundaries
    pattern = rf'\b{re.escape(word)}\b'

    # Search for the pattern in the path
    match = re.search(pattern, path)

    # Return True if a match is found, otherwise False
    return bool(match)


def extract_meta_from_path(path: str) -> tuple[str, str]:
    """
    Takes the path and extracts meta information from it. It Extracts the month in format YYYY-MM and the camera
    station. edited: Oliver Grothum
    """
    mth = re.findall('\d{4}-\d{2}', path)
    if mth:
        mth = mth[0]
    else:
        mth = None

    station = [subdir for subdir in path.split('\\') if 'Station' in subdir][0]

    return mth, station


def copy_files(source_file: str, dest_file: str, pbar: tqdm.tqdm, overwrite_existing=False, overwrite_size=False):
    if os.path.exists(dest_file) and not overwrite_existing and not overwrite_size:
        pbar.set_postfix_str(f'Skip: {os.path.basename(source_file)}')
        return
    elif overwrite_size:
        pbar.set_postfix_str(f'Overwrite (No matching Size): {os.path.basename(source_file)}')
        shutil.copy(source_file, dest_file)
    else:
        pbar.set_postfix_str(f'Copy: {os.path.basename(source_file)}')
        shutil.copy(source_file, dest_file)


if __name__ == '__main__':

    source_dir: str = r'D:\Grothum\Kamerstation_Metashape'  # local (source) directory
    dest_dir: str = r'Z:\000_Daten\archive\slope\1_processed'  # remote directory

    # Get all dense and filtered
    dense: list = [dns for dns in glob.glob(os.path.join(source_dir, '**', '*.ply'), recursive=True)]
    filtered: list = [flt for flt in glob.glob(os.path.join(source_dir, '**', '*.txt'), recursive=True) if
                      check_word_in_path(flt, 'filtered')]
    pt_prec: list = [pt for pt in glob.glob(os.path.join(source_dir, '**', '*.txt'), recursive=True) if
                     check_word_in_path(pt, 'ptPrecision')]
    m3c2: list = [pt for pt in glob.glob(os.path.join(source_dir, '**', '*.txt'), recursive=True) if
                  check_word_in_path(pt, 'm3c2')]
    timeseries: list = [file for file in glob.glob(os.path.join(source_dir, '**', '*.*'), recursive=True) if
                        check_word_in_path(file, 'timeseries')]

    # Overwriting files, if exist?
    overwrite_existing = False

    # dictionary to rename station name to english version
    rename_station: dict = {'Station_oben': 'station_top', 'Station_mitte': 'station_mid',
                            'Station_unten': 'station_bot'}

    # set empty to skip
    # dense = []
    # filtered = []
    # pt_prec = []
    # m3c2 = []
    # timeseries = []
    # iterate over dense and update them
    for dns in (pbar := tqdm.tqdm(dense, unit='files', position=0, leave=True)):
        dns_filename = os.path.basename(dns)
        pbar.set_postfix_str(dns_filename)
        mth, station = extract_meta_from_path(dns)
        station_en = rename_station[station]  # rename station directory into destinations english version

        dest_file = os.path.join(dest_dir, station_en, mth, 'Dense', dns_filename)
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)

        # check if filesize is equal, if not, set to overwrite
        overwrite_size = False
        if os.path.exists(dest_file) and os.path.getsize(dns) != os.path.getsize(dest_file):
            overwrite_size = True

        # copy file
        copy_files(dns, dest_file, pbar, overwrite_existing, overwrite_size)

    # iterate over filtered and update them
    for dns_filt in (pbar := tqdm.tqdm(filtered, unit='files', position=0, leave=True)):
        dns_filename = os.path.basename(dns_filt)
        pbar.set_postfix_str(dns_filename)
        mth, station = extract_meta_from_path(dns_filt)
        station_en = rename_station[station]  # rename station directory into destinations english version

        dest_file = os.path.join(dest_dir, station_en, mth, 'Dense', 'filtered', dns_filename)
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)

        # check if filesize is equal, if not, set to overwrite
        overwrite_size = False
        if os.path.exists(dest_file) and os.path.getsize(dns_filt) != os.path.getsize(dest_file):
            overwrite_size = True

        copy_files(dns_filt, dest_file, pbar, overwrite_existing, overwrite_size)

    # iterate over precision maps and update them
    for pt in (pbar := tqdm.tqdm(pt_prec, unit='files', position=0, leave=True)):
        pt_filename = os.path.basename(pt)
        pbar.set_postfix_str(pt_filename)
        mth, station = extract_meta_from_path(pt)
        station_en = rename_station[station]  # rename station directory into destinations english version

        dest_file = os.path.join(dest_dir, station_en, mth, 'ptPrecision', pt_filename)
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)

        # check if filesize is equal, if not, set to overwrite
        overwrite_size = False
        if os.path.exists(dest_file) and os.path.getsize(pt) != os.path.getsize(dest_file):
            overwrite_size = True

        copy_files(pt, dest_file, pbar, overwrite_existing, overwrite_size)

    # iterate over m3c2 resulst and update them
    for m3c2_file in (pbar := tqdm.tqdm(m3c2, unit='files', position=0, leave=True)):
        pt_filename = os.path.basename(m3c2_file)
        pbar.set_postfix_str(pt_filename)
        mth, station = extract_meta_from_path(m3c2_file)
        station_en = rename_station[station]  # rename station directory into destinations english version

        dest_file = os.path.join(dest_dir, station_en, 'm3c2', pt_filename)
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)

        # check if filesize is equal, if not, set to overwrite
        overwrite_size = False
        if os.path.exists(dest_file) and os.path.getsize(m3c2_file) != os.path.getsize(dest_file):
            overwrite_size = True

        copy_files(m3c2_file, dest_file, pbar, overwrite_existing, overwrite_size)

    for ts_file in (pbar := tqdm.tqdm(timeseries, unit='files', position=0, leave=True)):
        pt_filename = os.path.basename(ts_file)
        pbar.set_postfix_str(pt_filename)
        mth, station = extract_meta_from_path(ts_file)
        station_en = rename_station[station]  # rename station directory into destinations english version

        dest_file = os.path.join(dest_dir, station_en, 'm3c2', 'timeseries', pt_filename)
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)

        # check if filesize is equal, if not, set to overwrite
        overwrite_size = False
        if os.path.exists(dest_file) and os.path.getsize(ts_file) != os.path.getsize(dest_file):
            overwrite_size = True

        copy_files(ts_file, dest_file, pbar, overwrite_existing, overwrite_size)


