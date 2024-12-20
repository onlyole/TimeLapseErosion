from __future__ import annotations # for type annotations befor python 3.9

import os
import numpy as np
import cv2 as cv



class EllipseFit:

    def __init__(self, img_path: str, quality_thresh: float = 2):
        self.img_path = img_path
        self.binary_methode = cv.THRESH_BINARY+cv.THRESH_OTSU
        self.quality_thresh = quality_thresh
        self.rootdir_fit_result = ''
        self.dir_fit_result = 'ellipse_output'

        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None
        self.xc = None
        self.yc = None

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def refine_marker_with_ellipse_fit(self) -> tuple[tuple[float, float], float]:
        img = cv.imread(self.img_path, 1)

        # create roi of marker based on bounding box from cnn
        roi = img[self.y1:self.y2, self.x1:self.x2]
        roi_grey = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        # gaussian blurring of roi
        roi_blur = cv.GaussianBlur(roi_grey, (5,5), 0)
        # thresholding
        roi_binary, roi_thresh = cv.threshold(roi_blur, 0, 255, self.binary_methode)
        # finding contours

        contours, _ = cv.findContours(roi_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        image_center = (roi.shape[1] // 2, roi.shape[0] // 2)
        contour_centers = []
        for contour in contours:
            try:
                moments = cv.moments(contour)
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
                contour_center = (center_x, center_y)
                contour_centers.append(contour_center)
            except:
                continue
        sorted_contours = [c for c in contours if cv.pointPolygonTest(c, image_center, False) > 0]

        if not sorted_contours:
            # no contours found, leave and only return ai result
            return (self.xc, self.yc), self.quality_thresh + 1
        
        # Draw the most central contour
        central_contour = sorted_contours
        ellipse = cv.fitEllipse(central_contour[0])
        ellipse_center = (float(ellipse[0][0]) + self.x1, float(ellipse[0][1]) + self.y1)

        #TODO check if contour and ellipse overlaps

        quality = self._evaluate_ellipse_fit(central_contour[0], ellipse)

        # print(quality)
        if self.rootdir_fit_result:
            self._export_ellipse_fit_image(img, roi, central_contour, ellipse, quality)

        if self.quality_thresh and quality > self.quality_thresh:
            return (self.xc, self.yc), quality
        else:
            return ellipse_center, quality

    def _export_ellipse_fit_image(self, img, roi, central_contour, ellipse, quality):
        
        export_path = os.path.join(self.rootdir_fit_result, self.dir_fit_result)

        if not os.path.exists(export_path):
            os.makedirs(export_path)

        cv.drawContours(roi, central_contour[0], -1, (0, 0, 255), 2)
        cv.circle(roi, (round(ellipse[0][0]), round(ellipse[0][1])), 2, (0, 0, 255), 2)
        cv.ellipse(roi, ellipse, (0, 255, 0), 2)
            

            # result_path = rf'{dir_fit_result}/{os.path.basename(img_path).split(".")[0]}_result.jpg'
        result_path = os.path.join(export_path, f'{os.path.basename(self.img_path).split(".")[0]}_result.jpg')
        if os.path.exists(result_path):
            img_result = cv.imread(result_path, 1)
        else:
            img_result = img.copy()

        img_result[self.y1:self.y1+roi.shape[0], self.x1:self.x1+roi.shape[1]] = roi
        cv.putText(img_result, "quality: {:.4f}".format(quality), org=(self.x1, self.y1),
                       fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=2 ,color=(255,255,0), thickness=2)
            

        cv.imwrite(result_path, img_result)

    def _evaluate_ellipse_fit(self, contour, ellipse) -> float:

        ellipse_points = cv.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])),
                                        (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
                                        int(ellipse[2]), 0, 360, 1)

        contour = contour.reshape(-1, 2)

        distances = []
        for point in ellipse_points.tolist():
            
            dist = cv.pointPolygonTest(contour, tuple(point), True)
            distances.append(abs(dist))

        quality = np.max(distances)
        # quality = np.mean(distances)

        return quality