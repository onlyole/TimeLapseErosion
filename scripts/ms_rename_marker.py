import Metashape
import pandas as pd

doc = Metashape.app.document
for chunk in doc.chunks:
    for marker in chunk.markers:

        if len(marker.label) == 2:
            if 'A' in marker.label:
                marker.label = marker.label.replace('A', '10')
            elif 'B' in marker.label:
                marker.label = marker.label.replace('B', '20')
            elif 'C' in marker.label:
                marker.label = marker.label.replace('C', '30')
            elif 'D' in marker.label:
                marker.label = marker.label.replace('D', '40')
        elif len(marker.label) == 3:
            if 'A' in marker.label:
                marker.label = marker.label.replace('A', '1')
            elif 'B' in marker.label:
                marker.label = marker.label.replace('B', '2')
            elif 'C' in marker.label:
                marker.label = marker.label.replace('C', '3')