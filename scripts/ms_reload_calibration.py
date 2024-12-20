import Metashape
import pandas as pd

# load calibration_path_file
paths_file = r"D:\Grothum\Kamerastationen\calibration_paths_2022.txt"
df = pd.read_csv(paths_file, header=None, names=['cam', 'path'], sep=',')

doc = Metashape.app.document
for chunk in doc.chunks:
    images = chunk.cameras
    for img in images:
        # get camera name
        cam_name = img.group.label
        selected_row = df[df['cam'] == cam_name]
        if selected_row.shape[0] == 0:
            continue
        calibration_path = selected_row.iloc[0,1]
        sensor = img.sensor
        calib = Metashape.Calibration()
        calib.width = sensor.width
        calib.height = sensor.height
        calib.load(calibration_path, format=Metashape.CalibrationFormatXML)
        sensor.user_calib = calib

    chunk.optimizeCameras(fit_f= True,
                          fit_cx=True,
                          fit_cy=True,
                          fit_b1=False,
                          fit_b2=False,
                          fit_k1=False,
                          fit_k2=False,
                          fit_k3=False,
                          fit_k4=False,
                          fit_p1=False,
                          fit_p2=False,
                          fit_corrections=False,
                          adaptive_fitting=False,
                          tiepoint_covariance=False)