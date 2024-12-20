import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List

class PointNet2D:

    def __init__(self, label_points: list, marker_coords: pd.DataFrame):
            """
            Initializes an object of the class with a list of label points and a Pandas DataFrame of marker coordinates.

            Args:
                label_points (list): A list of tuples containing the label name, x-coordinate, and y-coordinate.
                marker_coords (pd.DataFrame): A Pandas DataFrame containing the marker coordinates with 'xc' and 'yc'
                    columns.
            """

            self.label_list = label_points
            self.precise_coords = marker_coords.copy(deep=True)

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def fetch_marker_label(self, dist_thresh=60, count_thresh=4):
        """
        Fetches the label for a marker based on its coordinates and a list of reference labels.

        Args:
            self: The object instance.
            dist_thresh (int, optional): The maximum distance threshold between the marker and a reference label.
                Defaults to 100.
            count_thresh (int, optional): The minimum number of similar points required for a label to be considered
                as the best match. Defaults to 3.

        Returns:
            self.precise_coords

        Prints:
            - If a label with the required number of similar points is found, it prints the label and the number of similar
            points.
            - If a label with a distance from the marker within a certain threshold is found, it prints the label and the
            distance.
            - If no label satisfies the above conditions, it prints a message indicating that no label was found.

        Raises:
            None.
        """
            
        for idx, row in self.precise_coords.iterrows():
            # calculate 2d net on measured marker
            marker_net = self.calc_net_on_query_point(float(row['xc']), float(row['yc']), self.precise_coords[['xc', 'yc']].values.tolist())
            label_value_count = {}
            label_value_std = {}
            #iterate thoru
            for item in self.label_list:
                    
                # calculate 2d net on label_net
                label_net = self.calc_net_on_query_point(item[3], item[4], [[point[3], point[4]] for point in self.label_list])
                
                count_sim_points, std_dist = self.compare_net(marker_net, label_net, thresh=dist_thresh)    
                label_value_count[item[2]] = (count_sim_points)
                label_value_std[item[2]] = (std_dist)
                
                if False:
                # if 'IMG_5707' in row['img']:
                    # pass
                    plt.plot([x[0] for x in marker_net], [x[1] for x in marker_net], linestyle='', marker='o')
                    plt.plot([x[0] for x in label_net], [x[1] for x in label_net], linestyle='', marker ='x')
                    plt.plot(item[3], item[4], linestyle='', marker='d')
                    plt.title(f'point {item[2]}, count_sim_points: {count_sim_points}, sigma_dist: {std_dist}')
                    plt.show()
            
            best_label_for_count = max(label_value_count, key=label_value_count.get)
            best_label_count = label_value_count[best_label_for_count]
            best_label_for_std = min(label_value_std, key=label_value_count.get)
            best_label_std = label_value_count[best_label_for_std]

            if best_label_count >= count_thresh:
                if 'IMG_1805' in row['img']:
                    print(f'highest value is {best_label_for_count} with {best_label_count} and {best_label_std}')
                self.precise_coords.at[idx, 'label'] = str(best_label_for_count)
            # elif best_label_std <= 10:
            #     # print(f'possible value is {best_label_for_std} with {best_label_std}')
            #     self.precise_coords.at[idx, 'label'] = str(best_label_for_std)
            else:
                # print(f'for {best_label_for_count}, {best_label_count} is to low')
                self.precise_coords.at[idx, 'label'] = ''

        return self.precise_coords

    def calc_net_on_query_point(self, query_x: float, query_y: float, net_points: list):

        query_net = sorted([(math.dist([query_x, query_y],[row[0], row[1]]), 
                            math.atan2(row[1] - query_y, row[0] - query_x)) for row in net_points])
        
        return [(math.cos(item[1])*item[0], math.sin(item[1])*item[0]*(-1)) for item in query_net]
    
    def compare_net(self, net_1: list, net_2: list, thresh=30):
        """Compares two point sets with its corepoint and other points relativ to its corepoint. 
        If two points are close to each other below a given threshold, they are assumed to be the same point.
        Returns the number of same point pairs.

        Args:
            net_1 (list): _description_
            net_2 (list): _description_
            thresh (int, optional): _description_. Defaults to 30.

        Returns:
            _type_: _description_
        """
        point_comparisons = []
        point_relation = []
        # for item1 in net_1:
        #     distances = sorted([math.dist([item1[0], item1[1]],[item2[0], item2[1]]) for item2 in net_2])
        #     point_comparisons.append(distances[0])


        for idx_1, item1 in enumerate(net_1):
            distances = [math.dist([item1[0], item1[1]],[item2[0], item2[1]]) for item2 in net_2]
            min_dist = min(distances)
            min_dist_idx = distances.index(min_dist)
            point_comparisons.append(min_dist)
            point_relation.append((idx_1, min_dist_idx))

        c = len([i for i in point_comparisons if i < thresh])
        m = np.array(point_comparisons).mean()
        s = (np.array(point_comparisons) - m).std()
        #print(f'{c} points appeal to be in same place of netshape')

        return c, s
            
        
if __name__ == '__main__':

    # kamera 11 - good kamera position
    # label_points = [["Kamera11",30,3140.6249999999973,3601.562499999997],
    #                ["Kamera11",32,673.8281249999994,2076.171874999998],
    #                ["Kamera11",33,3152.3437499999973,2076.171874999998],
    #                ["Kamera11",34,3283.2031249999973,1195.3124999999989],
    #                ["Kamera11",35,1599.6093749999986,1150.390624999999],
    #                ["Kamera11",36,1199.9999999999989,382.81249999999966],
    #                ["Kamera11",37,2324.9999999999977,446.8749999999996],
    #                ["Kamera11",38,3393.749999999997,459.3749999999996],
    #                ["Kamera11",39,5004.687499999995,490.62499999999955]]

    # marker_df = pd.DataFrame(
    #        np.array([['cam','img',3285.95386,1187.76101],
    #                  ['cam','img',1599.28231,1152.10636],
    #                  ['cam','img',5003.27061,488.759330],
    #                  ['cam','img',3143.18524,3600.52081],
    #                  ['cam','img',1197.74157,379.608040],
    #                  ['cam','img',675.954620,2075.50845],
    #                  ['cam','img',3394.33194,456.235160]]), columns=['cam', 'img', 'xc', 'yc'])
    
    # moved camera position

    label_points = [["Kamera11",30,3238.281249999999,3619.140624999999],
                    ["Kamera11",32,789.0624999999998,2089.8437499999995],
                    ["Kamera11",33,3236.328124999999,2085.9374999999995],
                    ["Kamera11",34,3367.187499999999,1199.2187499999998],
                    ["Kamera11",35,1699.2187499999995,1175.7812499999998],
                    ["Kamera11",36,1300.7812499999998,414.0624999999999],
                    ["Kamera11",37,2417.9687499999995,472.6562499999999],
                    ["Kamera11",38,3480.468749999999,472.6562499999999],
                    ["Kamera11",39,5095.703124999999,464.8437499999999]]

    marker_df = pd.DataFrame(
           np.array([['cam','img',3027.26555,3627.95264],
                     ['cam','img',3173.58084,1218.32233],
                     ['cam','img',1465.39474,1167.66797],
                     ['cam','img',4845.26486,509.03041],
                     ['cam','img',520.19193,2099.76666],
                     ['cam','img',3273.63573,487.46792],
                     ['cam','img',1058.24085,388.82478],
                     ['cam','img',3037.04425,2099.68658]]), columns=['cam', 'img', 'xc', 'yc'])    
    
    marker_df = marker_df.astype({'xc':'float','yc':'float'})
    print(marker_df.dtypes)

    net = PointNet2D(label_points, marker_df)
    df = net.fetch_marker_label(dist_thresh=75, count_thresh=3)
    
    print(df.head(10))

    pass