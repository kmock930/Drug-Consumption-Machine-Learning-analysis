import numpy as np;
import matplotlib.pyplot as plt;
import shap;
import random;

def plot(X: np.ndarray, y: np.ndarray, SHAP_values: list | np.ndarray, columns: list | np.ndarray, baseVal: float | int = None, index_samples: int = None, title: str = "", plotType: str = "summary"):
    uniqueLabels = np.unique(y);

    if index_samples is not None:
        index_samples = min(index_samples, X.shape[0]);
        random_indices = np.random.choice(X.shape[0], index_samples, replace=False);
        X_sample = X[random_indices];
        SHAP_values_sample = SHAP_values[random_indices];
    else:
        X_sample = X;
        SHAP_values_sample = SHAP_values;
    
    for class_index, class_name in enumerate(uniqueLabels):
        plot_title = f"{title} - class '{class_name}'" if title else f"Class '{class_name}'";
        plt.title(plot_title);
                
        if (SHAP_values.ndim == 3):
            shap_values_for_class = SHAP_values_sample[:, :, class_index];
        elif (SHAP_values.ndim == 2):
            shap_values_for_class = SHAP_values_sample;
        else:
            raise ValueError("Unexpected SHAP_values dimensions");
    
        match (plotType):
            case "summary":
                shap.summary_plot(
                    shap_values_for_class,
                    X_sample,
                    feature_names=columns
                );
                plt.show();
            case "force":
                if (baseVal == None):
                    raise ValueError("Missing Base (Expected) Value.");

                for i in range(index_samples):
                    shap.force_plot(
                        baseVal, 
                        shap_values_for_class[i, :], 
                        X_sample[i, :], 
                        feature_names=columns,
                        matplotlib=True
                    );
                    plt.title(title)
                    plt.show()
            