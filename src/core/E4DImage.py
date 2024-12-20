from PIL import Image, ExifTags
import exifread


class E4DPicture:

    def __init__(self, path_img):
        self._img_path = path_img
        self._img = Image.open(path_img)
        self._camera_name = ""
        self._camera_srn = ""
        self._img_datetime= ""

        self.search_key_in_exif()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._img = None

    def search_key_in_exif(self):

        exif = open(self._img_path, 'rb')

        tags = exifread.process_file(exif)

        for tag in tags.keys():
            if "Image Model" in tag:
                self._camera_name = tags[tag]
            if "BodySerialNumber" in tag:
                self._camera_srn = tags[tag]
            if "DateTimeOriginal" in tag:
                self._img_datetime = tags[tag]

    def get_camera_name(self):
        return self._camera_name

    def get_camera_srn(self):
        return self._camera_srn

    def get_img_datetime(self):
        return self._img_datetime