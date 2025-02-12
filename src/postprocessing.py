"""The postprocessing module contains a function to aggregate the RUL predicted within the same trajectory."""
from typing import Callable
import numpy as np
import pandas as pd
from src.constants import AGG_LAST, AGG_METHOD, AGG_LAST2

def align(y_pred: np.ndarray, cycles: np.ndarray, trajectory_id: np.ndarray):
    df = pd.DataFrame({'pred' : y_pred, 'Cycle' : cycles, 'TrajectoryID': trajectory_id})
    df['RUL'] = df['pred'] + df['Cycle'] - df.groupby('TrajectoryID')['Cycle'].transform('max')
    return df

def aggregate(y_pred: pd.DataFrame, y: np.ndarray, N_last: int = AGG_LAST, N_last2: int = AGG_LAST2, method: Callable[[int], float | int] = AGG_METHOD):
    RULs = []
    y_pred['truth'] = y
    agg_truths = []
    for _tid, result in y_pred.groupby('TrajectoryID'):
        results_overall =  result['RUL'].iloc[-N_last2:].values
        weights = method(np.arange(len(results_overall)) + N_last - len(results_overall) + 1)
        rul = np.mean(np.sum(results_overall*weights)/np.sum(weights))
        if rul < 40:
            results_to_be_considered = result['RUL'].iloc[-N_last:].values
            weights = method(np.arange(len(results_to_be_considered)) + N_last - len(results_to_be_considered) + 1)
            RULs.append(np.sum(results_to_be_considered*weights)/np.sum(weights))
        else:
            RULs.append(rul)
        agg_truths.append(result['truth'].min())
    return np.array(RULs), np.array(agg_truths)