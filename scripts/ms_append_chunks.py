import Metashape
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog

doc = Metashape.app.document

root = tk.Tk()
root.withdraw()

projects = filedialog.askopenfilenames(filetypes=[("Metashape Project file", "*.psx")])

for proj in projects:

    temp_doc = Metashape.Document()
    temp_doc.open(proj)

    proj_name = os.path.splitext(os.path.basename(proj))[0]

    for chunk in temp_doc.chunks:
        chunk.label = f"{proj_name}_{chunk.label}"

    doc.append(temp_doc)
