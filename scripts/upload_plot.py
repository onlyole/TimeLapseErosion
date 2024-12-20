import os
import pandas
import re
import glob

if __name__ == '__main__':
    ...
    source_root_path = r"F:\Bodenerosion4D\Beregnung\Daten"

    experiment_paths = [path for path in glob.glob(os.path.join(source_root_path, '*')) if os.path.isdir(path) and
                        re.findall("\d{1,2}_\d{6}", string=path)]

    for exp in experiment_paths:
        date = re.findall("\d{6}", exp)[0]
        date = f"20{date[4:6]}-{date[2:4]}-{date[0:2]}"

        exp_root_path = os.path.join(source_root_path, exp)

        copy_dirs = [dir for dir in glob.glob(os.path.join(exp_root_path, "**", "*"), recursive=True)
                     if any(ext in dir.lower() for ext in ["dense", "m3c2", "ptPrecision"]) and os.path.isdir(dir)
                     and not ".files" in dir]
        pass