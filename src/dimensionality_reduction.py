import pandas as pd
from src.constants import BR_THRESHOLD, PCA_N, PCA_PATH, SCALER_PATH


def backward_regression(df: pd.DataFrame, threshold_out: float = BR_THRESHOLD, verbose: bool = True):
    """
    Compute a backward regression on the inuput dataframe.

    Params:
    ---
    - df: pandas.DataFrame, the Dataframe containing the features and the target of the regression.
    - threshold: float, the minimum pvalue we consider to remove a column
    - verbose: bool, flag for printing the current state of the algorithm.

    Returns:
    ---
    columns_to_remove: set, the least informative columns of the DataFrame X regarding the regression target y. 
    """
    import statsmodels.api as sm

    X = df.drop(columns=['TrajectoryID', 'Cycle', 'RUL'], errors='ignore')
    y = df['RUL']

    columns_to_remove = set()
    included = list(X.columns) # The list of column included
    while True:
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out: # We remove the worst column
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            columns_to_remove.add(worst_feature)
            if verbose:
                print(f"- {worst_feature} removed with p_value of {worst_pval:.4f}")
        else:
            break
    if verbose:
        print(f"{columns_to_remove} are the columns to remove for a backward regression with threshold of {threshold_out}.")
    return columns_to_remove

def pca(df: pd.DataFrame, fit: bool = False, n_components: float = PCA_N, path: str = PCA_PATH):
    
    df_to_pca = df.drop(columns=['TrajectoryID', 'RUL', 'Cycle', 'Mach Number', 'Altitude', 'Sea Level Temperature'])
    from sklearn.decomposition import PCA
    import joblib
    import os
    if os.path.isfile(path):
        pca_: PCA = joblib.load(path)
        if pca_.n_components_ != n_components:
            pca_ = PCA(n_components)
    else:
        pca_ = PCA(n_components)
    
    if fit:
        pcs = pca_.fit_transform(df_to_pca)
    else:
        pcs = pca_.transform(df_to_pca)
    
    joblib.dump(pca_, path)

    new_df = pd.DataFrame(pcs, columns=[f"pc_{i+1}" for i in range(n_components)], index=df.index)
    new_df[['TrajectoryID', 'RUL', 'Cycle', 'Mach Number', 'Altitude', 'Sea Level Temperature']] = df[['TrajectoryID', 'RUL', 'Cycle', 'Mach Number', 'Altitude', 'Sea Level Temperature']]
    
    return new_df, pca_.explained_variance_
    