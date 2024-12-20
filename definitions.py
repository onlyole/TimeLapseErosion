# definitions for that project

import os, sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(ROOT_DIR, 'demo')
CLOUDCOMPY_DIR = os.path.join(ROOT_DIR, 'third_party', 'cloudcompy', 'CloudComPy310_20230705', 'CloudComPy310',
                              'CloudCompare')

cloudcompy_cc_path = CLOUDCOMPY_DIR
if cloudcompy_cc_path not in sys.path:
    sys.path.append(cloudcompy_cc_path)
