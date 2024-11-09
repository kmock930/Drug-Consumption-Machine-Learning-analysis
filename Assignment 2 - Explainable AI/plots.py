import numpy as np;
import matplotlib.pyplot as plt;
import shap;
import random;

def plot(X: np.ndarray, y: np.ndarray, SHAP_values: list | np.ndarray, columns: list | np.ndarray, baseVal: float | int = None, index_samples: int = None, title: str = "", plotType: str = "summary"):
    '''
    This is a function which plots the SHAP values.

    Parameters
    ------------
    X : numpy.ndarray - The data used in an explanation (usually the samples of the test set).
    y : numpy.ndarray - The target values.
    SHAP_values : list | numpy.ndarray - The SHAP values.
    columns : list | numpy.ndarray - The column names.
    baseVal : float | int - The base value.
    index_samples : int - The number of samples to be plotted.
    title : str - The title of the plot.
    plotType : str - The type of plot to be displayed.

    Returns
    ------------
    None

    Exceptions
    ------------
    ValueError : Invalid plotType. Please choose from 'summary', 'force', 'dependence', 'waterfall', 'decision'.
    ValueError : Unexpected SHAP_values dimensions.
    ValueError : Missing Base (Expected) Value.
    ValueError : Missing or Invalid Type of explainer! Please instantiate again.
    ValueError : Missing model or training data. Cannot evaluate the base value.
    ValueError : Missing or Invalid Type of explainer! Please instantiate again.

    Author
    ------------
    Kelvin Mock
    '''
    uniqueLabels = np.unique(y);


    if index_samples is not None:
        random_indices = random.sample(range(X.shape[0]), index_samples);
        X_sample = X[random_indices, :];
        SHAP_values_sample = SHAP_values[random_indices, :];
    else:
        X_sample = X;
        SHAP_values_sample = SHAP_values;
    
    for class_index, class_name in enumerate(uniqueLabels):
        plot_title = f"{title} - class '{class_name}'" if title else f"Class '{class_name}'";
                
        if (SHAP_values_sample.ndim == 3):
            shap_values_for_class = SHAP_values_sample[:, :, class_index]; # (n_samples, n_features, class_index)
        elif (SHAP_values_sample.ndim == 2):
            shap_values_for_class = SHAP_values_sample;
        else:
            raise ValueError("Unexpected SHAP_values dimensions");
        
        match (plotType):
            case "summary":
                plt.title(plot_title);
                shap.summary_plot(
                    shap_values_for_class,
                    X_sample,
                    feature_names=columns
                );
                plt.show();
            case "force":
                shap.force_plot(
                    baseVal[class_index] if isinstance(baseVal, np.ndarray) or isinstance(baseVal, list) else baseVal, 
                    shap_values_for_class[index_samples-1], # (n_samples, n_features)
                    X_sample[index_samples-1, :], # (n_samples-1, n_features)
                    feature_names=columns,
                    matplotlib=True,
                    show=False
                );
                # Adjust the scale of the plot
                ax = plt.gca()
                ax.xaxis.set_major_locator(plt.MultipleLocator(0.005));
                ax.xaxis.set_minor_locator(plt.MultipleLocator(0.001));
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'));

                plt.title(plot_title, y=1.75);
                plt.show();
            case "dependence":
                shap.dependence_plot(
                    SHAP_values=shap_values_for_class[index_samples, :], # (n_samples, n_features)
                    X=X_sample[index_samples, :],
                    interaction_index=None,
                    feature_names=columns
                );
                plt.title(title);
                plt.show();
            case "waterfall":
                shap.waterfall_plot(
                    baseVal[class_index] if isinstance(baseVal, np.ndarray) or isinstance(baseVal, list) else baseVal,
                    shap_values_for_class[index_samples, :],
                    feature_names=columns
                );
                plt.title(title);
                plt.show();
                if ((isinstance(baseVal, np.ndarray) or isinstance(baseVal, list)) == False):
                    break;
            case "decision":
                shap.decision_plot(
                    baseVal[class_index] if isinstance(baseVal, np.ndarray) or isinstance(baseVal, list) else baseVal,
                    shap_values_for_class[index_samples, :],
                    feature_names=columns
                );
                plt.title(title);
                plt.show();
                if ((isinstance(baseVal, np.ndarray) or isinstance(baseVal, list)) == False):
                    break;
            case _:
                raise ValueError("Invalid plotType. Please choose from 'summary', 'force', 'dependence', 'waterfall', 'decision'.");