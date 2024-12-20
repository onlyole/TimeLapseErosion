import pandas as pd
import csv
import os
import glob

class Marker:

    def __init__(self):
        self.marker_frame: pd.DataFrame = None
        self.marker_label: pd.DataFrame = None

    
    def import_marker_from_dir(self, dirpath, camname):

        data = []
        for name in glob.glob(rf'{dirpath}\*.txt'):
            file = open(name, 'r')
            d = list(csv.reader(file, delimiter=' '))
            d = [[camname] + [os.path.splitext(os.path.basename(name))[0]] + row for row in d]
            file.close()
            data.extend(d)

        if self.marker_frame is None:
            self.marker_frame = pd.DataFrame(data, columns=['cam', 'img', 'xc', 'yc', 'p%', 'x1', 'y1', 'x2', 'y2'], dtype=str)
        else:
            self.marker_frame = pd.concat([self.marker_frame,
                                          pd.DataFrame(data,
                                                       columns=['cam', 'img', 'xc', 'yc', 'p%', 'x1', 'y1', 'x2', 'y2'])])
        
        self.marker_frame['cam'] = self.marker_frame['cam'].values.astype(str)
        self.marker_frame['img'] = self.marker_frame['img'].values.astype(str)
        self.marker_frame['xc'] = self.marker_frame['xc'].values.astype(float)
        self.marker_frame['yc'] = self.marker_frame['yc'].values.astype(float)
        self.marker_frame['p%'] = self.marker_frame['p%'].values.astype(float)
        self.marker_frame['x1'] = self.marker_frame['x1'].values.astype(float)
        self.marker_frame['y1'] = self.marker_frame['y1'].values.astype(float)
        self.marker_frame['x2'] = self.marker_frame['x2'].values.astype(float)
        self.marker_frame['y2'] = self.marker_frame['y2'].values.astype(float)


    def import_marker_label_positions(self, filename):
        self.marker_label = pd.read_csv(filename, names=['cam', 'img', 'marker_label', 'x', 'y'], dtype=str)
        self.marker_label['cam'] = self.marker_label['cam'].values.astype(str)
        self.marker_label['img'] = self.marker_label['img'].values.astype(str)
        self.marker_label['marker_label'] = self.marker_label['marker_label'].values.astype(str)
        self.marker_label['x'] = self.marker_label['x'].values.astype(float)
        self.marker_label['y'] = self.marker_label['y'].values.astype(float)


if __name__ == '__main__':
    m = Marker()
    m.import_marker_from_dir(r'D:\Bodenerosion4D\CameraStation\3DReconstruction\Station_oben\2021-05\marker')