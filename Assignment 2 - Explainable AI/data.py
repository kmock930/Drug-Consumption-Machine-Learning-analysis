import numpy as np;
from sklearn.preprocessing import StandardScaler, LabelEncoder;

def normalize(y: np.ndarray):
    '''
    Normalize the labels array.
    '''
    le = LabelEncoder(); # convert categorical labels into numeric representation
    y = le.fit_transform(y);
    return y;