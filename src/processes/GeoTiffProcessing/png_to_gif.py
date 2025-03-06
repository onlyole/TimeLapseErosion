# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 09:17:47 2022

@author: Oliver
"""
import glob

import imageio
import os
import tkinter as tk
from tkinter import filedialog

import natsort
from tqdm import tqdm

if __name__ == '__main__':

    years = [str(x) for x in range(2020, 2024)]
    station = "Station_oben"
    # station = "Station_mitte"
    # station = "Station_unten"

    root_dir = rf"D:\Grothum\Kamerstation_Metashape\{station}\m3c2"
    output_dir = os.path.join(root_dir, 'timeseries')

    png_dir = natsort.natsorted([x for x in glob.glob(os.path.join(root_dir, 'geotiff', '*')) if os.path.isdir(x)])[-1]

    files = glob.glob(os.path.join(png_dir, '*.png'))

    for year in tqdm(years):

        temp_files = natsort.natsorted([x for x in files if year in x])

        ims = [imageio.v2.imread(f) for f in temp_files]

        outpath = os.path.join(output_dir, f'timegif_{year}.gif')
        imageio.mimwrite(outpath, ims, fps=10)
        # imageio.mimwrite('prag_field_v4.gif', ims, fps=10)
