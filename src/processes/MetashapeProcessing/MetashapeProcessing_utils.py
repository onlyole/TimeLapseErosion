import argparse


def string_list(arg):
    return arg.split(' ')


def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--month', type=str, action='store')
    parser.add_argument('--station', type=str, action='store')
    parser.add_argument('--data_root', type=str, action='store')
    parser.add_argument('--proc_root', type=str, action='store')
    parser.add_argument('--gcp_ref_file', type=str, action='store')
    parser.add_argument('--calib_path_file', type=str, action='store')
    parser.add_argument('--mode', type=str, choices=['all', 'daily', 'event'], action='store')
    parser.add_argument('--refine_labels', type=string_list, action='store')
    parser.add_argument('--checkpoint_labels', type=string_list, action='store')
    parser.add_argument('--reset_project', action='store_true')
    parser.add_argument('--import_pictures', action='store_true')
    parser.add_argument('--import_marker_coordinates', action='store_true')
    parser.add_argument('--import_reference', action='store_true')
    parser.add_argument('--disable_blured_img', action='store_true')
    parser.add_argument('--align_chunks', action='store_true')
    parser.add_argument('--refine_alignment', action='store_true')
    parser.add_argument('--dense_chunks', action='store_true')
    parser.add_argument('--export_dense', action='store_true')
    # parse arguments
    args = parser.parse_args()

    return args
