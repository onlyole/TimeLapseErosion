# %%
'''fetch floating license from server for metashape processing'''

import os, sys
from definitions import DEMO_DIR
from src.processes.MetashapeProcessing.MetashapeProcessing_utils import string_list, set_parser

# os.environ['agisoft_LICENSE'] = "C:\\Program Files\\Agisoft\\Metashape Pro\\rlm_roam.lic"
os.environ['RLM_LICENSE'] = "C:\\Program Files\\Agisoft\\Metashape Pro\\rlm_roam.lic"
sys.path.insert(1, r'/MetashapeProcessing')

from src.core.MetashapeProcessing import MetashapeProcessing
import logging

if __name__ == '__main__':

    # uncomment to use logger infos
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)

    # Configuration
    args = set_parser()

    args.month = string_list("2020-06")  # months to be processed
    args.station = 'Station_oben'  # experiment name
    args.data_root = rf"{DEMO_DIR}\e4d_demo\Kamerastationen"  # path to root dir of experiment directory (arg.station)
    args.proc_root = rf"{DEMO_DIR}\e4d_demo\Kamerstation_Metashape"  # path, where results are processed to
    args.gcp_ref_file = "PP_Koo_all_2020-08-11.txt"  # gcp file for labeled markers
    args.calib_path_file = rf"{DEMO_DIR}\e4d_demo\Kamerastationen\calibration_paths.txt"  # text file of metashape calibration location
    args.mode = "daily"  # processing mode (daily, event, all)
    args.refine_labels = string_list("40 41 44 45 46 47 48 49")  # gcp labels to refine with ellipse fit
    args.checkpoint_labels = string_list("42 43")  # gcp labels, which will be set to check points

    reset = False  # reset and clear metashape project
    first_iteration = True  # import all images chunk wise, import marker coordinates and first photo alignment
    reload_reference = True  # reload reference gcp file
    refinement = True  # Does iterative refinement of alignment, exports precision maps from 3D tie points
    dense = True  # start MVS densification and export dense fles

    cam_list = {
        "Station_oben": ['Kamera21', 'Kamera22', 'Kamera23', 'Kamera24', 'Kamera25'],
        "Station_mitte": ['Kamera11', 'Kamera12', 'Kamera13', 'Kamera14', 'Kamera15'],
        "Station_unten": ['Kamera1', 'Kamera2', 'Kamera3', 'Kamera4', 'Kamera5']
    }

    # Processing
    if reset:
        args.reset_project = True
    if first_iteration:
        args.import_pictures = True
        args.import_marker_coordinates = True
        args.import_reference = True
        args.disable_blured_img = True
        args.align_chunks = True
    if reload_reference:
        args.import_reference = True
    if refinement:
        args.refine_alignment = True
    if dense:
        args.dense_chunks = True
        args.export_dense = True

    for mth in args.month:

        # Set variables from arguments
        month = args.month
        station = args.station
        mode = args.mode
        root_data = args.data_root
        root_proc = args.proc_root
        gcp_reference_file = os.path.join(root_data, args.gcp_ref_file)
        calib_path_file = args.calib_path_file
        labels_to_refine = args.refine_labels
        checkpoint_labels = args.checkpoint_labels
        reset_project = args.reset_project
        import_pictures = args.import_pictures
        import_marker_coordinates = args.import_marker_coordinates
        import_reference = args.import_reference
        disable_blured_img = args.disable_blured_img
        align_chunks = args.align_chunks
        refine_alignment = args.refine_alignment
        dense_chunks = args.dense_chunks
        export_dense = args.export_dense

        # create MetashapeProcessing Object and import time attribution file
        proc = MetashapeProcessing()
        time_attr_file = os.path.join(root_data, station, f"Time_Attribution_{station}_{mth}.txt")

        # set attribution
        if mode == 'all':
            print("choosing all")
            proc.project_path = os.path.join(root_proc, station, mth, f'{mth}_all.psx')
            proc.set_image_attribution(time_attr_file, proc.InputImageMode.Daily)
        elif mode == 'daily':
            print("choosing daily")
            proc.project_path = os.path.join(root_proc, station, mth, f'{mth}_daily.psx')
            proc.set_image_attribution(time_attr_file, proc.InputImageMode.Daily)
        elif mode == 'event':
            print("choosing event")
            proc.project_path = os.path.join(root_proc, station, mth, f'{mth}_events.psx')
            proc.set_image_attribution(time_attr_file, proc.InputImageMode.Events)
        else:
            raise ValueError("Selecting mode not defined.")

        # setup project
        proc.read_camera_calibration(calib_path_file)
        proc.setup_project()
        proc.set_cameras(cam_list[station])

        #import images
        for cam in cam_list[station]:
            proc.import_marker_coordinates(os.path.join(root_data, station, mth, cam, 'result', 'gcp'),cam)
        proc.import_marker_label(os.path.join(root_data, station, mth, 'label_position.txt'))

        if reset_project:
            proc.reset_project()
        if import_pictures:
            proc.import_time_slices()
        chunks = proc.doc.chunks

        if import_marker_coordinates:
            proc.import_marker_measurements(*chunks, refine_labels=labels_to_refine, dir_fit_result='')
            proc.clean_insufficient_marker(*chunks)

        if import_reference:
            proc.import_marker_reference(*chunks, reference_file=gcp_reference_file, delimiter='\t')
            proc.set_check_marker(*chunks, check_marker_label=checkpoint_labels)

        if disable_blured_img:
            proc.disable_bad_images(*chunks, img_threshold=0.3)

        if align_chunks:
            proc.align_chunk(*chunks, skip_aligned=False)

        if refine_alignment:
            proc.refine_alignment(*chunks)

        proc.mark_error_chunks(*chunks, threshold_2d=2.0, threshold_3d=0.005)
        proc.save_project()

        if dense_chunks:
            proc.dense_chunk(*chunks)
        proc.save_project()

        if export_dense:
            proc.export_dense_clouds(*chunks)

        proc.export_dense_clouds(*chunks)

