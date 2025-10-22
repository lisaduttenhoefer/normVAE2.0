import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict

from utils.config_utils_model import Config_2D


def log_and_print(message: str):
    logging.info(message)
    print(message)

def log_and_print_test(message, logger=None, level=logging.INFO):
    
    print("----------[INFO]",message)
    if logger:
        if level == logging.INFO:
            logger.info(message)
        elif level == logging.WARNING:
            logger.warning(message)
        elif level == logging.ERROR:
            logger.error(message)
        elif level == logging.DEBUG:
            logger.debug(message)

def setup_logging(config: Config_2D):
    
    #Set up logging for the training session

    filename = os.path.join(
        config.LOGGING_DIR, f"{config.TIMESTAMP}_{config.RUN_NAME}_training_log_.txt"
    )
    
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("Starting new training session")
    logging.info(f"Configuration:")
    for parameter, value in config.__dict__.items():
        logging.info(f"{parameter}: {value}")

    title = f"Starting Catatonia CPA training session {config.RUN_NAME} at {config.TIMESTAMP}"
    print("")
    print("#" * 100)
    print("")
    print(title)
    print("-" * len(title))
    print("")
    print(f"Using Device:        {config.DEVICE}")
    print(f"Queued Epochs:       {config.TOTAL_EPOCHS}")
    print(f"Batch Size:          {config.BATCH_SIZE}")
    print(f"Data Summary:        {config.TRAIN_CSV}")
    print(f"MRI Data Directory:  {config.MRI_DATA_PATH}") #sagt jetzt immer train auch wenn eig test 
    print(f"Loading Model:       {config.LOAD_MODEL}")
    print(f"Output Directory:    {config.OUTPUT_DIR}")
    print("")
    print(f"More details in log file: {filename}")
    print("")
    print("#" * 100)
    
def setup_logging_test(log_file=None, log_level=logging.INFO):
    
    #Set up logging configuration
    
    logger = logging.getLogger("normative_vae")
    logger.setLevel(log_level)
    logger.handlers = []  
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_atlas_mode(atlas_name: str, num_rois: int):
    log_and_print(f"Using atlas: {atlas_name}\n"
                  f"Number of ROIs: {num_rois}")
    print("---------------")

def log_data_loading(datasets: Dict[str, int]):

    log_and_print("Data Processing")
    print("---------------")
    colon = ":"

    log_and_print("Loading Data")
    for dataset, n_subjects in datasets.items():
        log_and_print(
            f"  {dataset+colon:<20} {n_subjects:>4} subjects loaded"
        )
    print()


def log_model_setup():
    log_and_print("Creating Model")
    print("-" * len("Creating Model"))


def log_model_ready(model):

    log_and_print(f"Model Archtecture: \n{model}")
    print()

    colon = ":"

    for parameter, value in model.__dict__.items():
        if parameter in [
            "optimizer",
            "scheduler",
            "scaler",
            "contr_loss_weight",
            "kldiv_loss_weight",
            "recon_loss_weight",
            "class_loss_weight",	
            "adver_loss_weight",
            "contrast_temperature",
            "num_epochs",
            "device",
            "learning_rate",
            "weight_decay",
            "kernel_size",
            "stride",
            "padding",
            "image_size",
            "out_channels",
            "encoded_image_dim",
            "encoder_output_dim",
            "latent_dim",
            "schedule_on_validation",
            "scheduler_patience",
            "scheduler_factor",
            "dropout_prob",
            "use_ssim",
        ]:
            log_and_print(f"    {parameter+colon:<20} {str(value).splitlines()[0]}")

    # Logging total number of parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log_and_print(f"    Total Parameters:    {total_params:,}")
    log_and_print(f"    Trainable Params:    {trainable_params:,}")


def log_training_start():
    message = "Starting Training / Validation Loop"

    print()
    log_and_print(message)
    print("-" * len(message))


def log_model_metrics(
    epoch: int,
    metrics: Dict[str, float],
    type: str,
    prefix: str = "",
    n_digits: int = 3,
):

    metrics_log = ""

    for i, (metric, value) in enumerate(metrics.items()):
        metric += ":"
        significance = max(n_digits - len(str(int(value))), 0)

        if value < 10000:
            combo = f"{metric:<13} {value:>4.{significance}f},"
        else:
            combo = f"{metric:<13} {value:>7.1e},"

        metrics_log += f"{combo:<20}"
        # Print and reset every 4 metrics
        if i % 4 == 3:
            metrics_log = metrics_log[: metrics_log.rfind(",")]

            logging.info(f"Epoch {epoch:>3} - {metrics_log}")
            print(
                f"{prefix}{datetime.now()+timedelta(hours=2):%H:%M} - {type:<21}{metrics_log}"
            )

            metrics_log = ""


def log_extracting_latent_space(data_type: str):
    log_and_print(
        f"{datetime.now()+timedelta(hours=2):%H:%M} - Extracting Latent Space of {data_type}."
    )


def log_checkpoint(
    figure_path: str = None, model_path: str = None, metrics_path: str = None
):
    if figure_path is not None:
        log_and_print(
            f"{datetime.now()+timedelta(hours=2):%H:%M} - Checkpoint Save:     Figures saved to {figure_path}"
        )
    if model_path is not None:
        log_and_print(
            f"{datetime.now()+timedelta(hours=2):%H:%M} - Checkpoint Save:     Model saved to {model_path}"
        )
    if metrics_path is not None:
        log_and_print(
            f"{datetime.now()+timedelta(hours=2):%H:%M} - Checkpoint Save:     Metrics saved to {metrics_path}"
        )


def log_early_stopping(
    current_lr: float,
    min_lr: float,
    current_epoch: int,
):
    message = f"Stopping Training at Epoch {current_epoch} because current learning rate {current_lr} is below minimum learning rate {min_lr}."
    
    print()
    print("-"*len(message))
    log_and_print(message)
    print("-" * len(message))
    print()


def end_logging(config: Config_2D):
    end_messsage = f"session completed at {datetime.now()+timedelta(hours=2)}."

    logging.info(end_messsage)
    logging.info(f"Configuration:")
    for parameter, value in config.__dict__.items():
        logging.info(f"{parameter}: {value}")
    logging.info("End of training session")

    print("")
    print("#" * 100)
    print("")
    print(end_messsage)
    print("")
    print(f"Configuration:")
    for parameter, value in config.__dict__.items():
        parameter = parameter + ":"
        print(f"    {parameter:<20} {value}")
    print("")

    rocket_art = r"""

    *      *  |    *
      *      / \     * 
    *    *  / _ \  *    *
           |.o '.|  *
      *    |'._.'|         done :) 
         * |     |  *   *
    *    ,'|  |  |`.   *
         / |  |  |  \    *
    *   |,-'--|--'-. |
    
    """
    print(rocket_art)
    print("")
    print("")