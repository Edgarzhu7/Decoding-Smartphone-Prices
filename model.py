import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.linear_model import Lasso
import statsmodels.api as sm
from helper import *
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score, roc_auc_score, average_precision_score, 
    confusion_matrix, recall_score
)
from scipy import stats
from sklearn.preprocessing import StandardScaler
import numpy.typing as npt
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, max_error
import numpy as np

# config = yaml.load(open("config.yaml"), Loader=yaml.SafeLoader)
seed = 42
np.random.seed(seed)

def get_regression(
    loss: str = "squared_error",
    penalty: str | None = None,
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: float = 0.1,
) -> LinearRegression | Ridge | Lasso | KernelRidge:
    """
    Return a regression model based on the given loss, penalty function,
    and regularization parameter C.

    Args:
        loss: Specifies the loss function to use (default: 'squared_error').
        penalty: The type of penalty for regularization ('l1', 'l2', or None for no penalty).
        C: Regularization strength parameter (default: 1.0).
        kernel: Kernel type for Kernel Ridge Regression (default: 'rbf').
        gamma: Kernel coefficient for Kernel Ridge Regression (default: 0.1).

    Returns:
        A regression model based on the specified arguments.
    """
    
    if loss == "squared_error":
        if penalty == "l1":
            # Lasso regression (L1 regularization)
            return Lasso(alpha=C, random_state=42)
        elif penalty == "l2":
            # Ridge regression (L2 regularization)
            return Ridge(alpha=C, random_state=42)
        else:
            # Ordinary Least Squares (no regularization)
            return LinearRegression()
    elif loss == "kernel_ridge":
        # Kernel Ridge Regression
        return KernelRidge(alpha=1/(2*C), kernel=kernel, gamma=gamma)
    else:
        raise ValueError("Invalid loss function specified")


def compute_metric(y_true, y_pred, metric):
    """
    Computes the specified regression metric.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        metric: The regression metric to compute.

    Returns:
        The computed metric value.

    Raises:
        ValueError: If an unknown metric is specified.
    """
    
    if metric == "mae":
        return mean_absolute_error(y_true, y_pred)
    elif metric == "mse":
        return mean_squared_error(y_true, y_pred)
    elif metric == "rmse":
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric == "r2":
        return r2_score(y_true, y_pred)
    elif metric == "adjusted_r2":
        n = len(y_true)
        p = y_pred.shape[1] if y_pred.ndim > 1 else 1  # Number of predictors
        r2 = r2_score(y_true, y_pred)
        return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
    elif metric == "explained_variance":
        return explained_variance_score(y_true, y_pred)
    elif metric == "max_error":
        return max_error(y_true, y_pred)
    elif metric == "mape":
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    else:
        raise ValueError(f"Unknown metric: {metric}")

def performance(
    reg_trained,  # Fitted regression model (e.g., LinearRegression, Ridge, etc.)
    X: np.ndarray,
    y_true: np.ndarray,
    metric: str = "mae",
    bootstrap: bool = True
) -> tuple[np.float64, np.float64, np.float64] | np.float64:
    """
    Calculates the performance metric as evaluated on the true values (y_true)
    versus the predicted values from reg_trained and X, using 1,000 bootstrapped
    samples of the test set if bootstrap is set to True. Otherwise, returns
    single sample performance as specified by the user.
    
    Args:
        reg_trained: A fitted instance of a regression estimator.
        X : (n,d) np.array containing features.
        y_true: (n,) np.array containing true target values.
        metric: A string specifying the performance metric (default: 'mae').
                Other options: 'mse', 'rmse', 'r2', 'adjusted_r2', etc.
        bootstrap: Whether to perform bootstrapping for performance estimation.
                   If True, 1,000 bootstraps will be used to calculate the
                   empirical 95% confidence interval.
    
    Returns:
        If bootstrap is True: the median performance and the empirical 95% confidence interval (in np.float64).
        If bootstrap is False: the performance based on the specified metric.
    """
    
    # Predict target values using the trained regression model
    y_pred = reg_trained.predict(X)

    if not bootstrap:
        # Single performance calculation without bootstrapping
        return compute_metric(y_true, y_pred, metric)

    # If bootstrap=True, perform bootstrapping
    bootstrap_scores = []
    n_bootstraps = 1000
    n_samples = len(y_true)

    for _ in range(n_bootstraps):
        # Resample with replacement
        X_resampled, y_resampled = resample(X, y_true, n_samples=n_samples, replace=True)
        y_pred_resampled = reg_trained.predict(X_resampled)

        # Calculate metric for this bootstrap sample
        score = compute_metric(y_resampled, y_pred_resampled, metric)
        bootstrap_scores.append(score)

    # Calculate the median and 95% confidence interval
    lower_bound = np.percentile(bootstrap_scores, 2.5)
    upper_bound = np.percentile(bootstrap_scores, 97.5)
    median_score = np.median(bootstrap_scores)

    return median_score, lower_bound, upper_bound

def cv_performance(
    reg,  # Regression model (e.g., LinearRegression, Ridge, etc.)
    X: np.ndarray,
    y: np.ndarray,
    metric: str = "mae",
    k: int = 5,
) -> tuple[float, float, float]:
    """
    Splits the data X and the target y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a regression model
    on all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for the regression
    model by averaging the performance across folds.
    
    Args:
        reg: an instance of a sklearn regression model.
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features.
        y: (n,) vector of continuous target values.
        k: the number of folds (default=5).
        metric: the performance metric (default='mae').
    
    Returns:
        A tuple containing (mean, min, max) 'cross-validation' performance across the k folds.
    """
        # Reset index in case X and y are pandas DataFrames or Series
    if isinstance(X, pd.DataFrame):
        X = X.reset_index(drop=True)
    if isinstance(y, pd.Series):
        y = y.reset_index(drop=True)

    kf = KFold(n_splits=k, shuffle=False)
    performance_scores = []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the regression model on the training data
        reg.fit(X_train, y_train)
        # Predict the target values for the test data
        y_pred = reg.predict(X_test)

        # Calculate the performance on the test data
        performance_score = compute_metric(y_test, y_pred, metric)
        performance_scores.append(performance_score)

    # Return the mean, min, and max performance scores across all fold splits
    performance_tuple = (
        np.mean(performance_scores), 
        np.min(performance_scores), 
        np.max(performance_scores)
    )
    
    return performance_tuple

def select_param_lasso(
    X: np.ndarray,
    y: np.ndarray,
    metric: str = "mae",
    k: int = 5,
    c_range: list[float] = [],
) -> float:
    """
    Sweeps different settings for the hyperparameter of Lasso regression,
    calculating the k-fold CV performance for each setting on X, y.
    
    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features.
        y: (n,) array of continuous target values.
        k: the number of folds (default=5).
        metric: the performance metric for which to optimize (default='mae',
                other options: 'mse', 'rmse', 'r2').
        alpha_range: a list of alpha values to be searched over (LASSO regularization strength).
    
    Returns:
        The hyperparameter (alpha) for a Lasso regression model that maximizes the
        average k-fold CV performance.
    """
    
    best_score = -np.inf
    best_c = None

    for c in c_range:
        # Initialize a Lasso regression model with the current c value
        reg = get_regression(loss="squared_error", penalty='l1', C=c)
        
        # Calculate the cross-validation performance
        mean_performance, min_performance, max_performance = cv_performance(reg, X, y, metric, k)
        
        # Update the best hyperparameter if a better score is found
        if mean_performance > best_score:
            best_score = mean_performance
            best_c = c

    return best_c


def q2f(X_train, y_train, feature_names, C=0.001):
    # Train L1-regularized  regression with C = 1.0
    reg = get_regression(penalty='l1', C=C, loss='squared_error')
    reg.fit(X_train, y_train)
    
    # Extract the learned coefficients
    coefficients = reg.coef_
    
    # Create a dataframe of coefficients and feature names
    coef_df = pd.DataFrame({
        'Feature Name': feature_names,
        'Coefficient': coefficients
    })

    # Find the 4 most positive and 4 most negative coefficients
    top_positive = coef_df.nlargest(4, 'Coefficient')
    top_negative = coef_df.nsmallest(4, 'Coefficient')
    
    return top_positive, top_negative

def q4b_compare_models(X_train, y_train, X_test, y_test, C=1000):
    # Define metrics to evaluate
    metric_list = [
        "mae",
        "mse",
        "rmse",
        "r2",
        "adjusted_r2",
        "explained_variance",
        "max_error",
        "mape"
    ]
    # Initialize dictionary to store results
    results = {"L1-regularized Regression": {}, "Ridge Regression": {}}

    # Logistic Regression model
    l1reg = get_regression(loss="squared_error", penalty='l1', C=C)
    l1reg.fit(X_train, y_train)
    for metric in metric_list:
        median, lower_ci, upper_ci = performance(l1reg, X_test, y_test, metric=metric)
        results["L1-regularized Regression"][metric] = (median, lower_ci, upper_ci)

    # Ridge Regression model
    reg_ridge = get_regression(loss="kernel_ridge", C=C)
    reg_ridge.fit(X_train, y_train)
    y_pred_ridge = reg_ridge.predict(X_test)
    
    # Convert continuous predictions to binary using threshold 0
    # y_pred_binary = np.where(y_pred_ridge >= 0, 1, -1)

    for metric in metric_list:
        median, lower_ci, upper_ci = performance(reg_ridge, X_test, y_test, metric=metric)
        results["Ridge Regression"][metric] = (median, lower_ci, upper_ci)

    # Display the results
    for model, metrics in results.items():
        print(f"\nModel: {model}")
        for metric, (median, lower_ci, upper_ci) in metrics.items():
            print(f"{metric.capitalize()}: Median = {median:.4f}, 95% CI = ({lower_ci:.4f}, {upper_ci:.4f})")



def q4d(X, y, C=1.0, gamma_range=[0.001, 0.01, 0.1, 1, 10, 100], metric="mae", k=5):
    # Use KFold for regression tasks
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Store the results in a dictionary
    results = {}

    for gamma in gamma_range:
        print(f"\nEvaluating model with gamma = {gamma}")
        scores = []
        
        # Reset index for consistency if using pandas DataFrame/Series
        if isinstance(X, pd.DataFrame):
            X = X.reset_index(drop=True)
        if isinstance(y, pd.Series):
            y = y.reset_index(drop=True)

        # Perform cross-validation
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train the Kernel Ridge Regression model
            reg = KernelRidge(alpha=1/(2*C), kernel='rbf', gamma=gamma)
            reg.fit(X_train, y_train)

            # Predict on the validation set
            y_pred = reg.predict(X_val)

            # Compute performance metric
            score = compute_metric(y_val, y_pred, metric)  # Remove y_pred as the 4th argument
            scores.append(score)

        # Calculate mean, min, and max scores
        mean_score = np.mean(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        results[gamma] = (mean_score, min_score, max_score)
        
        print(f"Gamma = {gamma}: Mean {metric} = {mean_score:.4f}, Min = {min_score:.4f}, Max = {max_score:.4f}")

    return results

def q4e(X_train, y_train, X_test, y_test, feature_names, C=0.001):
    reg = get_regression(loss="kernel_ridge", C=C, gamma=0.1)
    reg.fit(X_train, y_train)
    metric_list = [
        "mae",
        "mse",
        "rmse",
        "r2",
        "adjusted_r2",
        "explained_variance",
        "max_error",
        "mape"
    ]
    results = {}

    # Evaluate performance with bootstrapping
    for metric in metric_list:
        median, lower_ci, upper_ci = performance(reg, X_test, y_test, metric=metric, bootstrap=True)
        results[metric] = {
            "median": median,
            "95% CI": (lower_ci, upper_ci)
        }
    
    # Print results in table format
    print(f"Kernel Ridge Regression (C={0.001}, gamma={100}) Test Performance")
    print(f"{'Performance Measure':<20}{'Median Performance':<20}{'95% Confidence Interval'}")
    for metric, result in results.items():
        print(f"{metric.capitalize():<20}{result['median']:<20.4f}({result['95% CI'][0]:.4f}, {result['95% CI'][1]:.4f})")
    
    return results

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.utils import resample
import pandas as pd

# Assuming X and y are your features and target
def bootstrap_lasso(X, y, alpha, n_bootstraps=1000, random_state=42):
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    lasso = Lasso(alpha=alpha, random_state=random_state)
    coefficients = np.zeros((n_bootstraps, n_features))

    for i in range(n_bootstraps):
        # Resample the data
        X_resampled, y_resampled = resample(X, y)
        # Fit the LASSO model
        lasso.fit(X_resampled, y_resampled)
        # Store the coefficients
        coefficients[i, :] = lasso.coef_
    
    # Calculate confidence intervals (e.g., 2.5th and 97.5th percentiles for 95% CI)
    lower_bounds = np.percentile(coefficients, 2.5, axis=0)
    upper_bounds = np.percentile(coefficients, 97.5, axis=0)
    return lower_bounds, upper_bounds, np.mean(coefficients, axis=0)


def bootstrap_pvalues(X, y, alpha, n_bootstraps=1000, random_state=42):
    np.random.seed(random_state)
    lasso = Lasso(alpha=alpha, random_state=random_state)
    coefficients = []

    # Generate bootstrap samples and fit LASSO
    for _ in range(n_bootstraps):
        X_resampled, y_resampled = resample(X, y)
        lasso.fit(X_resampled, y_resampled)
        coefficients.append(lasso.coef_)

    coefficients = np.array(coefficients)

    # Calculate p-values: proportion of times coefficients are non-zero
    p_values = np.mean(coefficients != 0, axis=0)
    return p_values

def bootstrap_psi(X, y, alpha, n_bootstraps=1000):
    from sklearn.linear_model import Lasso
    from sklearn.utils import resample
    import numpy as np
    import statsmodels.api as sm

    lasso = Lasso(alpha=alpha, random_state=42)
    boot_coefs = []

    for _ in range(n_bootstraps):
        # Generate bootstrap resample
        X_resampled, y_resampled = resample(X, y)

        # Fit LASSO on bootstrap sample
        lasso.fit(X_resampled, y_resampled)
        selected_features = np.where(lasso.coef_ != 0)[0]

        if len(selected_features) > 0:
            # Refit OLS on selected features
            X_selected = X_resampled[:, selected_features]
            X_selected = sm.add_constant(X_selected)  # Add intercept
            ols = sm.OLS(y_resampled, X_selected).fit()
            coef_full = np.zeros(X.shape[1])  # Include zeros for non-selected
            coef_full[selected_features] = ols.params[1:]  # Skip intercept
            boot_coefs.append(coef_full)

    boot_coefs = np.array(boot_coefs)

    # Calculate mean coefficients
    mean_coefs = np.mean(boot_coefs, axis=0)
    if len(boot_coefs) == 0:
        raise ValueError("boot_coefs is empty. No valid bootstrap samples were collected.")
    # Calculate confidence intervals
    conf_intervals = np.percentile(boot_coefs, [2.5, 97.5], axis=0)
    
    # Calculate p-values
    p_values = np.mean(boot_coefs != 0, axis=0)

    # Calculate standard errors
    std_errors = np.std(boot_coefs, axis=0, ddof=1)  # ddof=1 for unbiased estimate

    return mean_coefs, conf_intervals, p_values, std_errors


def main() -> None:
    print(f"Using Seed={seed}")
    # Read data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_vector, impute_missing_values AND normalize_feature_matrix
    X_train, y_train, X_test, y_test, feature_names = get_train_test_split('2019.csv')
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    C_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    metric_list = [
        "mae",
        "mse",
        "rmse",
        "r2",
        "adjusted_r2",
        "explained_variance",
        "max_error",
        "mape"
    ]

    # #Q1.d
    # Calculate the required statistics
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    print(X_train_df.columns.tolist())
    summary_table = pd.DataFrame({
        'Feature': X_train_df.columns,
        'Mean': X_train_df.mean(),
        'Interquartile Range': X_train_df.quantile(0.75) - X_train_df.quantile(0.25),
    })
    # print(X_train_df)
    # # # Display the summary table
    # print(summary_table)

    # #Q21.b
    C_range = [0.001, 0.01, 0.1]
    performance_table = []

    for metric in metric_list:
        best_C = select_param_lasso(X_train, y_train, metric, 5, C_range)

        # print(f"Best C for {metric}: {best_C}")
        reg = get_regression(loss="squared_error", penalty='l1', C=best_C)
        mean_Perf, min_Perf, max_Perf = cv_performance(reg, X_train, y_train, metric, 5)
        performance_table.append((metric, best_C, 'l1', f"{mean_Perf:.4f} ({min_Perf:.4f}, {max_Perf:.4f})"))

    print(f"{'Performance Measures':<20} {'C':<10} {'Penalty':<10} {'Mean (Min, Max) CV Performance'}")
    for row in performance_table:
        print(f"{row[0]:<20} {row[1]:<10} {row[2]:<10} {row[3]}")

    # We want to minimize rmse with C=?
    reg = get_regression(loss="squared_error", penalty='l1', C=1)
    reg.fit(X_train, y_train)

    feature_names = X_train_df.columns.tolist()
    # print(feature_names)
    # print(reg.coef_) 
    # Set LASSO alpha and call the function
    assert X_train.shape[0] > 0, "X_train is empty!"
    assert y_train.shape[0] > 0, "y_train is empty!"
    assert not np.isnan(y_train).any(), "y_train contains NaN values!"
    assert not np.isnan(X_train).any().any(), "X_train contains NaN values!"

    # lower, upper, mean_coef = bootstrap_lasso(X_train, y_train, alpha=1.0, n_bootstraps=1000, random_state=42)
    mean_coef, conf_intervals, p_values, se = bootstrap_psi(X_train, y_train, alpha=0.1, n_bootstraps=1000)
    lower = conf_intervals[0, :]
    upper = conf_intervals[1, :]
    # Create a DataFrame for easy interpretation
    # feature_names = X_train.columns if hasattr(X_train, "columns") else [f"Feature_{i}" for i in range(X_train.shape[1])]
    feature_names = X_train_df.columns.tolist()
    coef_summary = pd.DataFrame({
        "Feature": feature_names,
        "Mean Coefficient": mean_coef,
        "Lower Bound (2.5%)": lower,
        "Upper Bound (97.5%)": upper,
        "P-Value": p_values,
        "Standard Error": se
    })
    print(coef_summary)
    # results = {}
    # for metric in metric_list:
    #     median, lower_ci, upper_ci = performance(reg, X_test, y_test, metric=metric)
    #     results[metric] = {
    #         "median": median,
    #         "95% CI": (lower_ci, upper_ci)
    #     }
    # print(f"C = {c}, Penalty = l1")
    # # print(f"C = 1, Penalty = l1")
    # for metric, result in results.items():
    #     print(f"{metric.capitalize()}: Median = {result['median']:.4f}, 95% CI = ({result['95% CI'][0]:.4f}, {result['95% CI'][1]:.4f})")
   
   # Q2.f
   # Get the top positive and negative coefficients
    # top_positive, top_negative = q2f(X_train, y_train, feature_names, C=1)

    # # Display the results
    # print("Top Positive Coefficients:")
    # print(top_positive)

    # print("\nTop Negative Coefficients:")
    # print(top_negative)

if __name__ == "__main__":
    main()
