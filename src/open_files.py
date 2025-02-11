import pandas as pd
from src.constants import TRAIN_FILES, TEST_FILES, RUL_FILES, TRAIN_VALIDATION_SPLIT, COLUMNS
def open_train_val(train_files: list[str] = TRAIN_FILES):
    from random import choices

    df = pd.DataFrame({c : [] for c in COLUMNS})

    for i, file in enumerate(train_files):
        file_df = pd.read_csv(
            file,
            sep=' ',
            header=None,
            names=COLUMNS,
            usecols=range(0, 26)
        )

        unique_trajs = file_df['TrajectoryID'].unique()
        val_trajs = choices(unique_trajs, k=int(len(unique_trajs)*TRAIN_VALIDATION_SPLIT))
        file_df['val'] = file_df['TrajectoryID'].apply(lambda tid: (tid in val_trajs)) # set a subset of the trajectories as train and another as validation
        file_df['TrajectoryID'] = file_df['TrajectoryID'].astype(str) + f'_{i+1}' # for unicity of the trajectoryIDs across several files
        file_df['RUL'] = (file_df.groupby('TrajectoryID')['Cycle'].transform('max') - file_df['Cycle'])

        df = pd.concat([df, file_df])
    
    df['RUL'] = df['RUL'].astype(int)
    df['val'] = df['val'].astype(bool)

    return df.drop(columns=['val'])[~df['val']], df.drop(columns=['val'])[df['val']]

def open_train(train_files: list[str] = TRAIN_FILES):
    df = pd.DataFrame({c : [] for c in COLUMNS})

    for i, file in enumerate(train_files):
        file_df = pd.read_csv(
            file,
            sep=' ',
            header=None,
            names=COLUMNS,
            usecols=range(0, 26)
        )

        file_df['TrajectoryID'] = file_df['TrajectoryID'].astype(str) + f'_{i+1}' # for unicity of the trajectoryIDs across several files
        file_df['RUL'] = (file_df.groupby('TrajectoryID')['Cycle'].transform('max') - file_df['Cycle'])

        df = pd.concat([df, file_df])
    
    df['RUL'] = df['RUL'].astype(int)

    return df


def open_test(test_files: list[str] = TEST_FILES, rul_files: list[str] = RUL_FILES):

    df = pd.DataFrame({c : [] for c in COLUMNS})

    for i, (test_file, rul_file) in enumerate(zip(test_files, rul_files)):
        file_df = pd.read_csv(
            test_file,
            sep=' ',
            header=None,
            names=COLUMNS,
            usecols=range(0, 26)
        )

        unique_trajs = file_df['TrajectoryID'].unique()
        rul_df = pd.read_csv(rul_file, names=['RUL'])
        rul_df = pd.DataFrame(data = {'RUL' : rul_df['RUL'], 'TrajectoryID' : unique_trajs})
        file_df['RUL'] = (file_df.groupby('TrajectoryID')['Cycle'].transform('max') - file_df['Cycle'])
        file_df = file_df.merge(rul_df, on='TrajectoryID', how='left', suffixes=('', '_end'))
        file_df['RUL'] += file_df['RUL_end']
        file_df['TrajectoryID'] = file_df['TrajectoryID'].astype(str) + f'_{i+1}'

        df = pd.concat([df, file_df])
    
    df['RUL'] = df['RUL'].astype(int)
    return df.drop(columns=['RUL_end'])
