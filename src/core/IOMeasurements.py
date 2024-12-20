from __future__ import annotations # for type annotations befor python 3.9

import pandas as pd
import Metashape
from src.core.PointNet2D import PointNet2D
from src.core.EllipseFit import EllipseFit

class IOMeasurements:

    def __init__(self, chunk: Metashape.Chunk, marker_positions: pd.DataFrame, marker_label: pd.DataFrame):

        self.marker_positions = marker_positions
        self.marker_label = marker_label
        self.chunk = chunk
        self.refine_ellipse_labels = []
        self.refine_ellipse_output_dir = ''

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def import_marker_measurements(self) -> None:
        
        for img in self.chunk.cameras:
            if not img.label:
                continue
            # select relevant markers for camera
            marker_df, marker_label = self._select_marker(img)

            # print(marker_label)
            if not marker_label: #list is empty
                continue

            #append label to marker df
            labeling_obj = PointNet2D(marker_label, marker_df)
            marker_df = labeling_obj.fetch_marker_label()

            # label marker after template coords
            for _, row in marker_df.iterrows():

                if not row['label']:
                    continue

                xc_apriori= row['xc']
                yc_apriori= row['yc']
                marker_found = False
                for mrk in self.chunk.markers:     
                    if row['label'] == mrk.label:
                        marker_found = True
                        self._set_projection(img, row, mrk, xc_apriori, yc_apriori)

                        break

                if not marker_found:
                    mrk = self.chunk.addMarker()
                    mrk.label = str(row['label'])

                    self._set_projection(img, row, mrk, xc_apriori, yc_apriori)

    def _set_projection(self, img, row, mrk, xc_apriori, yc_apriori):

        if any([mrk.label in x for x in self.refine_ellipse_labels]):
            with EllipseFit(img.photo.path) as ef:
                ef.x1 = round(float(row['x1']))
                ef.x2 = round(float(row['x2']))
                ef.y1 = round(float(row['y1']))
                ef.y2 = round(float(row['y2']))
                ef.xc = xc_apriori
                ef.yc = yc_apriori
                ef.rootdir_fit_result = self.refine_ellipse_output_dir

                (xc_aposteriori, yc_aposteriori), quality = ef.refine_marker_with_ellipse_fit()

            # (xc, yc), quality = self._refine_marker_with_ellipse_fit(img.photo.path, mrk.label, row, dir_fit_result=self.refine_ellipse_output_dir)
            mrk.projections[img] = Metashape.Marker.Projection(Metashape.Vector([xc_aposteriori, yc_aposteriori]), True)
                                    
        else:
            mrk.projections[img] = Metashape.Marker.Projection(Metashape.Vector([xc_apriori, yc_apriori]), True)

    def _select_marker(self, img: Metashape.Chunk.Camera) -> None:

        marker_df = self.marker_positions.copy()
        marker_df = marker_df[marker_df['cam'].str.contains(img.group.label)]
        marker_df = marker_df[marker_df['img'].str.contains(img.label.split('.')[0])]

        marker_label = self.marker_label.copy()
        marker_label = marker_label[marker_label['cam'].str.contains(img.group.label)].values.tolist()

        return marker_df,marker_label