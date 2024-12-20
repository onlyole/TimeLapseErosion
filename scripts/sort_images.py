import PIL.Image
import PIL.ExifTags
import os, glob
import pandas as pd
from collections import Counter
import pathlib
import tqdm
import shutil

""" Sorts images """

if __name__ == '__main__':

    source_dir = r'F:\einsortieren_hang_02'
    dest_dir = r'F:\Bodenerosion4D\Rohdaten\Rohdaten_Kamerastation\SLRs'

    # Get all images
    images = [img for img in glob.glob(os.path.join(source_dir, '**', '*.JPG'), recursive=True)]

    # Get all destination kamera directories
    cam_dirs = [cam for cam in glob.glob(os.path.join(dest_dir, '**'), recursive=True) if 'Kamera' in cam.split('\\')[-1] and os.path.isdir(cam)]
    camera_names = [f'Kamera{i}' for i in [1, 2, 3, 4, 5,
                                           11, 12, 13, 14, 15,
                                           21, 22, 23, 24, 25]]

    # check all images in all for serial number and check if unique
    camname_to_serial = []
    for cam in camera_names:
        cam_paths = [x for x in cam_dirs if cam == os.path.basename(x)]

        images = []
        for path in cam_paths:
            images.extend(pathlib.Path(path).glob('*.jpg'))

        serial_numbers = []
        image_and_time = []
        for img in tqdm.tqdm(images[::5], unit='images', desc=f'{cam}'):

            pic_load = PIL.Image.open(img)
            pic_exif = {PIL.ExifTags.TAGS[k]: v for k, v in pic_load._getexif().items() if k in PIL.ExifTags.TAGS}
            try:
                serial_numbers.append(pic_exif['BodySerialNumber'])
            except:
                serial_numbers.append(pic_exif['Model'])
            # image_and_time.append((pic, pd.to_datetime(pic_exif['DateTime'], format='%Y:%m:%d %H:%M:%S')))

        unique_keys = Counter(serial_numbers).keys()
        unique_counts = Counter(serial_numbers).values()

        camname_to_serial.append((cam, list(unique_keys)))

    # copy images to path

    source_path = pathlib.Path(source_dir)
    source_images = list(source_path.rglob('*.jpg'))

    exception_catch = 0
    image_copied = 0

    for img in tqdm.tqdm(source_images, unit='images', desc=f'Process images', position=1):
        # get exif
        pic_load = PIL.Image.open(img)
        pic_exif = {PIL.ExifTags.TAGS[k]: v for k, v in pic_load._getexif().items() if k in PIL.ExifTags.TAGS}

        # get datetime and transform to YYYY-MM
        img_month = pd.to_datetime(pic_exif['DateTime'], format='%Y:%m:%d %H:%M:%S').strftime("%Y-%m")
        # get serial number
        try:
            img_serial = pic_exif['BodySerialNumber']
        except:
            img_serial = pic_exif['Model']

        target_camera = None
        for pair in camname_to_serial:
            if any([img_serial in s for s in pair[1]]):
                target_camera = pair[0]
                continue

        if target_camera is None:
            ValueError("SerialNumber does not exist in storage")
        # get stationname
        if any([target_camera in s for s in [f"Kamera{x}" for x in range(0,6)]]):
            station_name = "Station_unten"
        elif any([target_camera in s for s in [f"Kamera{x}" for x in range(10,16)]]):
            station_name = "Station_mitte"
        elif any([target_camera in s for s in [f"Kamera{x}" for x in range(20,26)]]):
            station_name = "Station_oben"
        # set new path
        new_file = os.path.join(dest_dir, station_name, img_month, target_camera, os.path.basename(img))

        if not os.path.exists(os.path.dirname(new_file)):
            os.makedirs(os.path.dirname(new_file), exist_ok=True)

        try:
        #copy image
            if os.path.exists(new_file):
                raise FileExistsError(f"File {new_file} already exists.")

            shutil.copy(img, os.path.dirname(new_file))
            image_copied += 1
        except FileExistsError as e:
            print(e)
            exception_catch += 1
            continue

    print(f'{image_copied} Files copied, {exception_catch} Files not copied.')
