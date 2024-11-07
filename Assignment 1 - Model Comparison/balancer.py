import constants;
import numpy as np;
from imblearn.under_sampling import RandomUnderSampler;
from imblearn.over_sampling import SMOTE;
from imblearn.pipeline import Pipeline;

rus = RandomUnderSampler(
    sampling_strategy=constants.underSamplePercent,
    random_state=constants.random_state
);
smote = SMOTE(
    sampling_strategy=constants.oversampledPercent,
    random_state=constants.random_state
);

def underSample(X_train: np.ndarray, y_train: np.ndarray):
    return rus.fit_resample(X_train, y_train);

def overSample(X_train: np.ndarray, y_train: np.ndarray):
    return smote.fit_resample(X_train, y_train);

def combineSample(X_train: np.ndarray, y_train: np.ndarray):
    pipeline = Pipeline(steps=[('undersampling', rus), ('oversampling', smote)]);
    return pipeline.fit_resample(X_train, y_train);