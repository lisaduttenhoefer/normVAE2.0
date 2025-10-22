import os
import subprocess as sp
from datetime import datetime, timedelta
from typing import List
import torch


def get_free_gpu() -> torch.device:
    #Get the GPU with the most available memory,
    #Falls back to CPU if no GPU is available or if there's an error accessing the GPUs
    
    # check if cuda is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        return torch.device("cpu")
    try:
        num_gpus = torch.cuda.device_count()
        
        if num_gpus == 0:
            print("No CUDA devices found despite CUDA being available. Using CPU instead.")
            return torch.device("cpu")
        
        # If there's only one GPU, use it without further checks
        if num_gpus == 1:
            print(f"Only one CUDA device found. Using cuda:0")
            return torch.device("cuda:0")
        
        # Try to run nvidia-smi command to get memory info
        try:
            import subprocess as sp
            command = "nvidia-smi --query-gpu=memory.free --format=csv"
            memory_free_info = (
                sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
            )
            
            # extract memory values
            memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
            
            # Check if we have memory data for all GPUs
            if len(memory_free_values) != num_gpus:
                print(f"Warning: nvidia-smi reported {len(memory_free_values)} GPUs, but torch.cuda sees {num_gpus}.")
                print(f"Using the first available GPU to be safe.")
                return torch.device("cuda:0")
                
            # Get the GPU with the most free memory
            gpu_idx = memory_free_values.index(max(memory_free_values))
            print(f"Selected GPU {gpu_idx} with {memory_free_values[gpu_idx]} MB free memory")
            return torch.device(f"cuda:{gpu_idx}")
            
        except (sp.SubprocessError, ValueError, IndexError) as e:
            print(f"Error getting GPU memory info: {e}")
            print(f"Falling back to first available GPU")
            return torch.device("cuda:0")
            
    except Exception as e:
        print(f"Unexpected error selecting GPU: {e}")
        print("Falling back to CPU")
        return torch.device("cpu")

#======================================================================================================================================
#CONFIG CLASS DEFINITION (2D)
class Config_2D:

    def __init__(
        self,
        # The learning rate for the model optimizer
        LEARNING_RATE: float,
        # The weight decay for the model optimizer
        WEIGHT_DECAY: float,
        # The batch size of the data loaders
        BATCH_SIZE: int,
        # The total number of epochs to train the model for
        TOTAL_EPOCHS: int,
        # The weight of the reconstruction loss of the model
        RECON_LOSS_WEIGHT: float,
        # The weight of the KL divergence loss of the model
        KLDIV_LOSS_WEIGHT: float,
        # The latent space dimensionality of the model
        LATENT_DIM: int,
        # A string to identify your run (will be in name of the output directory)
        RUN_NAME: str,
        # The path to the csv files (contains metadata and filenames of MRI .nii files)
        TRAIN_CSV: List[str],
        # The paths to the csv files that contain metadata for the testing data
        TEST_CSV: List[str],
        # Name of atlas which should be used for training of the model
        ATLAS_NAME: List[str],
        # The path to the folder where intermediate normalitzed and scaled data should be saved
        PROC_DATA_PATH: str,
        # The path to the directory that contains the MRI .nii files
        MRI_DATA_PATH: str,
        # The folder in which a training specific output directory should be created
        OUTPUT_DIR: str,
        # The column names, from the CSVs, that contain the covariates that you want to be attached to the Subject objects
        # The list of diagnoses that you want to include in training. 
        DIAGNOSES: List[str],
        # The list of volume types you want to include (Vgm, Vwm, csf)
        VOLUME_TYPE: List[str],
        #The list of volume types that are known
        VALID_VOLUME_TYPES: List[str],
        # Whether to use the Structural Similarity Index (SSIM) as a loss function for reconstruction loss.
        USE_SSIM: bool = False,
        # Whether to use early stopping during training, based on the LR being too low.
        EARLY_STOPPING: bool = True,
        # The learning rate at which to stop training if early stopping is enabled.
        STOP_LEARNING_RATE: float = 1e-6,
        # The probability of dropout in each convolutional stack of the model. Not all models support this.
        DROPOUT_PROB: float = 0.1,
        # The weight of of the classifier loss. Not all models have classifiers.
        CLASS_LOSS_WEIGHT: float = None,
        # The weight of the contrastive loss. Only ContrastVAE uses this.
        CONTR_LOSS_WEIGHT: float = None,
        # The temperature of the contrastive loss. Temperature is a hyperparameter similar to a loss weight.
        # Only ContrastVAE uses this.
        CONTRAST_TEMPERATURE: float = None,
        # The path to the csv file containing the adversarial training data. Only AdverVAE uses this.
        ADVER_CSV: str = None,
        # The learning rate for the the adversarial training optimizer. Only AdverVAE uses this.
        ADVER_LR: float = None,
        # Whether to load a pre-trained model or not. If True, PRETRAIN_MODEL_PATH and PRETRAIN_METRICS_PATH must be set.
        LOAD_MODEL: bool = False,
        # The path to the pre-trained model to load.
        PRETRAIN_MODEL_PATH: str = None,
        # The path to the pre-trained model's performance metrics to load.
        PRETRAIN_METRICS_PATH: str = None,
        # The epoch to continue training from if loading a pre-trained model. If it is None, the model will start from the 0th epoch.
        CONTINUE_FROM_EPOCH: int = None,
        # The number of epochs between saving model checkpoints. Model checkpoints save the model's weights, latent representations, and latent plots.
        CHECKPOINT_INTERVAL: int = 10,
        # Wether to plot loss curves with a rolling window as well (can help clarify trends)
        METRICS_ROLLING_WINDOW: int = 10,
        # The size of the dots in the UMAP plot. Larger dots are easier to see, but may overlap.
        UMAP_DOT_SIZE: int = 30,
        # The number of neighbors to consider when plotting the UMAP plot. More neighbors can help clarify clusters, but hide local structure.
        UMAP_NEIGHBORS: int = 15,
        # Should training data be shuffled in the DataLoader?
        SHUFFLE_DATA: bool = True,
        # Seed for reproducibility
        SEED: int = 123,
        # The first training epochs can be volatile, and can mess with the scale of the plots. This parameter allows you to skip the first n epochs.
        DONT_PLOT_N_EPOCHS: int = 0,
        # Whether to schedule the learning rate based on the validation loss. If False, the learning rate will be scheduled based on the training loss.
        # Scheduling based on the validation loss can help prevent overfitting.
        SCHEDULE_ON_VALIDATION: bool = True,
        # The number of epochs to wait before reducing the learning rate if the loss does not improve.
        SCHEDULER_PATIENCE: int = 10,
        # The factor by which to reduce the learning rate if the loss does not improve.
        SCHEDULER_FACTOR: float = 0.01,
        # The device to use for training. If None, the device with the most free memory will be retrieved.
        DEVICE: torch.device = None,
        # A timestamp for the run. If None, the current time will be used. Timestamp will be in output directory name.
        TIMESTAMP: str = None,
        NUM_WORKERS = 2,  # Reduce if causing memory issues
        PIN_MEMORY = True,
        DROP_LAST = True,
        # Memory management settings
        CLEAR_CACHE_FREQUENCY = 10,  # Clear cache every N batches
        EMERGENCY_CLEANUP_ON_ERROR = True
    ):

        # set up mandatory training parameters
        self.LEARNING_RATE = LEARNING_RATE
        self.WEIGHT_DECAY = WEIGHT_DECAY
        self.BATCH_SIZE = BATCH_SIZE
        self.LATENT_DIM = LATENT_DIM
        self.CONTR_LOSS_WEIGHT = CONTR_LOSS_WEIGHT
        self.RECON_LOSS_WEIGHT = RECON_LOSS_WEIGHT
        self.KLDIV_LOSS_WEIGHT = KLDIV_LOSS_WEIGHT
        self.ADVER_LR = ADVER_LR
        self.CLASS_LOSS_WEIGHT = CLASS_LOSS_WEIGHT
        self.CONTRAST_TEMPERATURE = CONTRAST_TEMPERATURE
        self.TOTAL_EPOCHS = TOTAL_EPOCHS
        self.VOLUME_TYPE = VOLUME_TYPE
        self.VALID_VOLUME_TYPES = VALID_VOLUME_TYPES
        # self.COVARS = COVARS

        # set up optional training parameters
        self.CHECKPOINT_INTERVAL = CHECKPOINT_INTERVAL
        self.DIAGNOSES = DIAGNOSES
        self.UMAP_NEIGHBORS = UMAP_NEIGHBORS
        self.SHUFFLE_DATA = SHUFFLE_DATA
        self.SEED = SEED
        self.START_EPOCH = 0
        self.DROPOUT_PROB = DROPOUT_PROB
        self.USE_SSIM = USE_SSIM
        self.SCHEDULE_ON_VALIDATION = SCHEDULE_ON_VALIDATION
        self.SCHEDULER_PATIENCE = SCHEDULER_PATIENCE
        self.SCHEDULER_FACTOR = SCHEDULER_FACTOR
        self.BATCH_SIZE = BATCH_SIZE # Start small and increase if memory allows
        self.NUM_WORKERS = NUM_WORKERS  # Reduce if causing memory issues
        self.PIN_MEMORY = PIN_MEMORY
        self.DROP_LAST = DROP_LAST

        # Memory management settings
        self.CLEAR_CACHE_FREQUENCY = CLEAR_CACHE_FREQUENCY  # Clear cache every N batches
        self.EMERGENCY_CLEANUP_ON_ERROR = EMERGENCY_CLEANUP_ON_ERROR

        # training stop parameters
        self.EARLY_STOPPING = EARLY_STOPPING
        self.STOP_LEARNING_RATE = STOP_LEARNING_RATE

        if DEVICE is not None:
            self.DEVICE = DEVICE
        else:
            self.DEVICE = get_free_gpu()

        if TIMESTAMP is not None:
            self.TIMESTAMP = TIMESTAMP
        else:
            # server is off german local time by 2 hrs
            current_time = datetime.now() + timedelta(hours=2)
            self.TIMESTAMP = current_time.strftime("%Y-%m-%d_%H-%M")

        # set up plotting parameters
        self.DONT_PLOT_N_EPOCHS = DONT_PLOT_N_EPOCHS
        self.METRICS_ROLLING_WINDOW = METRICS_ROLLING_WINDOW
        self.UMAP_DOT_SIZE = UMAP_DOT_SIZE

        # set up run parameters ------------------------------------------------

        # set up model loading parameters for pre-trained models
        self.LOAD_MODEL = LOAD_MODEL

        if self.LOAD_MODEL:
            for path in [PRETRAIN_MODEL_PATH, PRETRAIN_METRICS_PATH]:
                assert (
                    path is not None
                ), "PRETRAIN_MODEL_PATH and PRETRAIN_METRICS_PATH must be set if LOAD_MODEL is True."
                assert os.path.exists(path), f"Path {path} does not exist."
            self.START_EPOCH = CONTINUE_FROM_EPOCH

        self.PRETRAIN_MODEL_PATH = PRETRAIN_MODEL_PATH
        self.PRETRAIN_METRICS_PATH = PRETRAIN_METRICS_PATH
        self.FINAL_EPOCH = self.START_EPOCH + self.TOTAL_EPOCHS

        # check that all other paths exist
        # for path in [ADVER_CSV, MRI_DATA_PATH_TRAIN, MRI_DATA_PATH_TEST, OUTPUT_DIR]:
        for path in [ADVER_CSV, MRI_DATA_PATH, OUTPUT_DIR]:
            if path is not None:
                assert os.path.exists(path), f"Path {path} does not exist"

        # set up paths
        self.OUTPUT_DIR = OUTPUT_DIR
        self.TRAIN_CSV = TRAIN_CSV
        self.ADVER_CSV = ADVER_CSV
        self.MRI_DATA_PATH = MRI_DATA_PATH
        self.PROC_DATA_PATH = PROC_DATA_PATH
        self.ATLAS_NAME = ATLAS_NAME
        self.TEST_CSV = TEST_CSV

        # set up path dependent parameters
        if RUN_NAME is None:
            self.RUN_NAME = os.path.split(self.TRAIN_CSV)[-1].split(".")[0]
        else:
            self.RUN_NAME = RUN_NAME

        self.FIGURES_DIR = os.path.join(self.OUTPUT_DIR, "figures")
        self.LOGGING_DIR = os.path.join(self.OUTPUT_DIR, "logs")
        self.DATA_DIR = os.path.join(self.OUTPUT_DIR, "data")
        self.MODEL_DIR = os.path.join(self.OUTPUT_DIR, "models")

    def __str__(self):
        return str(vars(self))