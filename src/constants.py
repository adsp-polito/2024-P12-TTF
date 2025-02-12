"""The constants module contains constants useful for the preprocessing"""

# The name of the features in the data files
COLUMNS = ['TrajectoryID', 'Cycle', 'Altitude', 'Mach Number',
           'Sea Level Temperature', 'T2', 'T24', 'T30', 'T50',
           'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30',
           'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed',
           'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']

# Preprocessing Hyper parameters:
WINDOW_SIZE = 19
WINDOW_STEP = 1
SAVGOL_WINDOW = 11
SAVGOL_POLYORDER = 1
TRAIN_VALIDATION_SPLIT = 0.2
SCALER_PATH = 'saved_models/scaler.skl'
SCALER_FULL_PATH = 'saved_models/full_scaler.skl'
FEATURES_TO_REMOVE = ['Nf', 'farB', 'NRc', 'P15', 'P30', 'T24', 'T50']

# Dimensionality reduction parameters:
BR_THRESHOLD = 0.1
PCA_N = 5
PCA_PATH = 'saved_models/pca.skl'

# Files
TRAIN_FILES = [f"data/train_FD00{i}.txt" for i in range(1, 5)]
TEST_FILES = [f"data/test_FD00{i}.txt" for i in range(1, 5)]
RUL_FILES = [f"data/RUL_FD00{i}.txt" for i in range(1, 5)]

# Post processing hyperparameters
AGG_METHOD = lambda t:t**0
AGG_LAST = 9 # For low ruls
AGG_LAST2 = 20 # For high ruls

# Transformers hyperparameters
BATCH_SIZE=128
D_MODEL=256
N_HEADS=16
N_LAYERS=4
FEEDFORWARD_DIM=128
DROPOUT=0.2

PRETRAIN_EPOCHS = 3
FINETUNING_EPOCHS = 100
PATIENCE = 15