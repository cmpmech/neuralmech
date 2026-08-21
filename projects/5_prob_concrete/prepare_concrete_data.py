import pandas as pd
import numpy as np

df = pd.read_excel('../../external_data/Concrete_Data.xls')

np.save('../../data/concrete_data.npy', df.values)
np.save('../../data/concrete_labels.npy', list(df))