import Metashape
import numpy as np
import pandas as pd
import math
import os, sys
import glob
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def calculate_marker_reprojection_error(chunk):
    """Calculate each 2D Projection and it's error of each marker in each photo"""

    marker_reproj_error = []
    marker_mean = []
    for marker in chunk.markers:
        if not marker.position:
            continue
        position = marker.position
        proj_error = []
        proj_sqsum = 0

        temp_marker_reproj_error = []
        for camera in marker.projections.keys():
            if not camera.transform:
                continue  # skipping not aligned cameras
            image_name = camera.label
            proj = marker.projections[camera].coord
            reproj = camera.project(marker.position)
            error = reproj - proj

            # print(f'{marker.label} - {camera.group} - {error.norm()}')

            temp_marker_reproj_error.append([camera.label, marker.label, error.norm()])

            proj_error.append(error.norm())
            proj_sqsum += error.norm() ** 2

        if len(proj_error):
            # 3sigma on reproj error

            repr_error = np.array([x[2] for x in temp_marker_reproj_error])
            #rmse = np.mean(repr_error, axis=0)
            rmse = math.sqrt(np.square(repr_error).mean())

            marker_mean.append([marker.label, rmse])

            error = math.sqrt(proj_sqsum / len(proj_error))
            # print(f"{marker.label} projection error: {mean}")
            marker_reproj_error.extend(temp_marker_reproj_error)

    return marker_reproj_error, marker_mean

def plot_graph(error, mean, header):
    x = [item[0] for item in error]
    y = [item[1] for item in error]

    x_mean = [item[0] for item in mean]
    y_mean = [item[1] for item in mean]

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.scatter(x_mean, y_mean, color='r')
    ax.set_title(header)
    ax.axhline(y=1.1, color='r', linestyle='--')
    if x:
        plt.xticks(range(1, max(x)+1))
    return fig


if __name__ == '__main__':

    # check lic
    os.environ['RLM_LICENSE'] = "C:\\Program Files\\Agisoft\\Metashape Pro\\rlm_roam.lic"

    lic = Metashape.License()
    if lic.valid:
        print("Found valid license")
    else:
        print("No license found")


    # get all metashape projects from root directory
    # metashape_path = [r"D:\Grothum\Kamerstation_Metashape\Station_unten\2020-06\2020-06_daily.psx",
    #                   r"D:\Grothum\Kamerstation_Metashape\Station_unten\2020-07\2020-07_daily.psx"]

    root = r"D:\Grothum\Kamerstation_Metashape\Station_unten"
    metashape_path = [x for x in glob.glob(os.path.join(root, '**', '*.psx'))]

    fig_collection = []
    for item in metashape_path:
        doc = Metashape.Document()
        doc.open(item)

        errors = []
        mean = []
        for idx, chunk in enumerate(doc.chunks):
            reproj_error, marker_mean = calculate_marker_reprojection_error(chunk)

            chunk_errors = []
            for row in reproj_error:
                chunk_errors.append(row[2])


            errors.extend([(idx + 1, x) for x in chunk_errors])
            mean.extend([(idx + 1, x[1]) for x in marker_mean])
            pass

        #plot scatter

        fig_collection.append(plot_graph(errors, mean, os.path.basename(item)))

    pp = PdfPages('foo.pdf')
    for f in fig_collection:
        pp.savefig(f)
    pp.close()
