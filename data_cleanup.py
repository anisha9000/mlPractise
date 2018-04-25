import numpy as np
import pandas as pd
import scipy.io as sio
meta_info = sio.loadmat('wiki.mat')
image_content = meta_info['wiki'].item(0)
required_data = np.column_stack((image_content[2][0], image_content[3][0]))
label = required_data[:, 1]
