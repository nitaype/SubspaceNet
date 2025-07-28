"""Subspace-Net main script 
    Details
    -------
    Name: main.py
    Authors: D. H. Shmuel
    Created: 01/10/21
    Edited: 30/06/23

    Purpose
    --------
    This script allows the user to apply the proposed algorithms,
    by wrapping all the required procedures and parameters for the simulation.
    This scripts calls the following functions:
        * create_dataset: For creating training and testing datasets 
        * training: For training DR-MUSIC model
        * evaluate_dnn_model: For evaluating subspace hybrid models

    This script requires that requirements.txt will be installed within the Python
    environment you are running this script in.

"""
# Imports
import sys
import torch
import os
import matplotlib.pyplot as plt
import warnings
from src.system_model import SystemModelParams
from src.signal_creation import *
from src.data_handler import *
from src.criterions import set_criterions
from src.training import *
from src.evaluation import test_dnn_model
from src.plotting import initialize_figures
from pathlib import Path
from src.models import ModelGenerator
import wandb
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
wandb.login(key="d55bad0e0b8a03b9bdfa4beeacf508cd29e1e398")
print("hello")

# Initialization
warnings.simplefilter("ignore")
os.system("cls||clear")
plt.close("all")

if __name__ == "__main__":
    # ------------------------------------------------------------------------------------
    # ---- Initialize time, date, simulation name and checkpoint name for evaluation ----
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    simulation_filename = "2speakers_full_data_doa_circular" + dt_string_for_save
    checkpoint = '2speakers_full_data_doa_circular16_07_2025_10_56'
    # checkpoint = '2speakers_full_data_23_06_2025_08_37'
    # ------------------------------------------------------------------------------------

    # Initialize paths
    external_data_path = Path("/gpfs0/bgu-br/users/tatarjit/model-based-nir/data")
    scenario_data_path = "diff_mvdr"
    datasets_path = external_data_path / "datasets" / scenario_data_path
    simulations_path = external_data_path / "simulations"
    saving_path = external_data_path / "weights_cluster"

    # create folders if not exists
    datasets_path.mkdir(parents=True, exist_ok=True)
    (datasets_path / "train").mkdir(parents=True, exist_ok=True)
    (datasets_path / "test").mkdir(parents=True, exist_ok=True)
    datasets_path.mkdir(parents=True, exist_ok=True)
    simulations_path.mkdir(parents=True, exist_ok=True)
    saving_path.mkdir(parents=True, exist_ok=True)
    # Operations commands
    commands = {
        "SAVE_TO_FILE": False,  # Saving results to file or present them over CMD
        "CREATE_DATA": False,  # Creating new dataset
        "LOAD_DATA": False,  # Loading data from exist dataset
        "LOAD_RECORDINGS": True,  # Load recordings from external file
        "LOAD_MODEL": False,  # Load specific model for training
        "TRAIN_MODEL": False,  # Applying training operation
        "SAVE_MODEL": True,  # Saving tuned model
        "EVALUATE_MODE": True,  # Evaluating desired algorithms
    }
    # Saving simulation scores to external file
    if commands["SAVE_TO_FILE"]:
        file_path = (
            simulations_path / "results" / "scores" / Path(dt_string_for_save + ".txt")
        )
        sys.stdout = open(file_path, "w")
    # Define system model parameters
    system_model_params = (
        SystemModelParams()
        .set_parameter("N", 15)
        .set_parameter("M", 1)
        .set_parameter("T", 157)
        .set_parameter("snr", 10)
        .set_parameter("signal_type", "NarrowBand")
        .set_parameter("signal_nature", "non-coherent")
        .set_parameter("eta", 0)
        .set_parameter("bias", 0)
        .set_parameter("sv_noise_var", 0)
    )
    # Generate model configuration
    model_config = (
        ModelGenerator()
        .set_model_type("SubspaceNet")
        .set_diff_method("music") # mvdr / music
        .set_tau(8)
        .set_model(system_model_params)
    )
    # Define samples size
    samples_size = 5000  # Overall dateset size
    train_test_ratio = 0.05  # training and testing datasets ratio
    # Sets simulation filename
    # simulation_filename = get_simulation_filename(
    #     system_model_params=system_model_params, model_config=model_config
    # )
    # Print new simulation intro
    print("------------------------------------")
    print("---------- New Simulation ----------")
    print("------------------------------------")
    print("date and time =", dt_string)
    # Initialize seed
    set_unified_seed()
    # Datasets creation
    if commands["CREATE_DATA"]:
        # Define which datasets to generate
        create_training_data = True  # Flag for creating training data
        create_testing_data = True  # Flag for creating test data
        print("Creating Data...")
        if create_training_data:
            # Generate training dataset
            train_dataset, _, _ = create_dataset(
                system_model_params=system_model_params,
                samples_size=samples_size,
                model_type=model_config.model_type,
                tau=model_config.tau,
                save_datasets=True,
                datasets_path=datasets_path,
                true_doa=None,
                phase="train",
            )
        if create_testing_data:
            # Generate test dataset
            test_dataset, generic_test_dataset, samples_model = create_dataset(
                system_model_params=system_model_params,
                samples_size=int(train_test_ratio * samples_size),
                model_type=model_config.model_type,
                tau=model_config.tau,
                save_datasets=True,
                datasets_path=datasets_path,
                true_doa=None,
                phase="test",
            )
    # Datasets loading
    elif commands["LOAD_DATA"]:
        (
            train_dataset,
            test_dataset,
            generic_test_dataset,
            samples_model,
        ) = load_datasets(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            samples_size=samples_size,
            datasets_path=datasets_path,
            train_test_ratio=train_test_ratio,
            is_training=True,
        )

    # Load recordings from external file
    elif commands["LOAD_RECORDINGS"]:
        # Load recordings from external file
        if model_config.diff_method == "mvdr":
            train_dataset = SimDS("/gpfs0/bgu-br/users/tatarjit/model-based-nir/2speakers_ds/si_tr_s_preprocessed")
            val_dataset = SimDS("/gpfs0/bgu-br/users/tatarjit/model-based-nir/2speakers_ds/si_dt_05_preprocessed")
            test_dataset = SimDS("/gpfs0/bgu-br/users/tatarjit/model-based-nir/2speakers_ds/si_et_05_preprocessed")
        if model_config.diff_method == "music":
            # --- 2 speakers with reverb ---
            # train_dataset = SimDSdoa("/gpfs0/bgu-br/users/tatarjit/model-based-nir/2speakers_ds_doa2/si_tr_s_preprocessed")
            # val_dataset = SimDSdoa("/gpfs0/bgu-br/users/tatarjit/model-based-nir/2speakers_ds_doa2/si_dt_05_preprocessed")
            # test_dataset = SimDSdoa("/gpfs0/bgu-br/users/tatarjit/model-based-nir/2speakers_ds_doa2/si_et_05_preprocessed")
            # train_dataset = SimDSdoa("/gpfs0/bgu-br/users/tatarjit/model-based-nir/2speakers_ds_doa2/2spkrs_1tau_0.5sec/train_preprocessed")
            # val_dataset = SimDSdoa("/gpfs0/bgu-br/users/tatarjit/model-based-nir/2speakers_ds_doa2/2spkrs_1tau_0.5sec/validation_preprocessed")
            # test_dataset = SimDSdoa("/gpfs0/bgu-br/users/tatarjit/model-based-nir/2speakers_ds_doa2/2spkrs_1tau_0.5sec/test_preprocessed")

            # --- one sine wave ---
            # train_dataset = SimDSdoa("/gpfs0/bgu-br/users/tatarjit/model-based-nir/1sin_ds/train_preprocessed")
            # val_dataset = SimDSdoa("/gpfs0/bgu-br/users/tatarjit/model-based-nir/1sin_ds/val/si_dt_05_preprocessed")
            # test_dataset = SimDSdoa("/gpfs0/bgu-br/users/tatarjit/model-based-nir/1sin_ds/test/si_et_05_preprocessed")

            # --- 1 speaker without reverb ---
            # train_dataset = SimDSdoa("/gpfs0/bgu-br/users/tatarjit/model-based-nir/1speaker_ds/si_tr_s_preprocessed")
            # val_dataset = SimDSdoa("/gpfs0/bgu-br/users/tatarjit/model-based-nir/1speaker_ds/si_dt_05_preprocessed")
            # test_dataset = SimDSdoa("/gpfs0/bgu-br/users/tatarjit/model-based-nir/1speaker_ds/si_dt_05_preprocessed")

            # --- 2 speakers without reverb ---
            train_dataset = SimDSdoa("/gpfs0/bgu-br/users/tatarjit/model-based-nir/2speakers_ds_noEcho/si_tr_s_preprocessed")
            val_dataset = SimDSdoa("/gpfs0/bgu-br/users/tatarjit/model-based-nir/2speakers_ds_noEcho/si_dt_05_preprocessed")
            test_dataset = SimDSdoa("/gpfs0/bgu-br/users/tatarjit/model-based-nir/2speakers_ds_noEcho/si_dt_05_preprocessed")

        print("Loaded recordings")

    # Training stage
    if commands["TRAIN_MODEL"]:
        # Assign the training parameters object
        simulation_parameters = (
            TrainingParams()
            .set_batch_size(16)
            .set_epochs(300)
            .set_model(model=model_config)
            .set_optimizer(optimizer="Adam", learning_rate=1e-3, weight_decay=1e-6)
            .set_training_dataset(train_dataset, val_dataset)
            .set_schedular(step_size=10, gamma=0.3, total_epochs=300, start_lr=1e-3, warmup_epochs=1, min_lr_factor=0.1)
            .set_criterion()
        )
        if commands["LOAD_MODEL"]:
            simulation_parameters.load_model(
                loading_path=saving_path / checkpoint
            )
        # Print training simulation details
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            parameters=simulation_parameters,
            phase="training",
        )
        # Initialize W&B for experiment tracking
        wandb.init(
            project="SubspaceNet",                     # Your W&B project name
            name=simulation_filename,                  # Unique run name
            config={
                "batch_size": simulation_parameters.batch_size,
                "epochs": simulation_parameters.epochs,
                "learning_rate": simulation_parameters.learning_rate,
                "warmup_epochs": simulation_parameters.warmup_epochs,
                "warmup_start_lr": simulation_parameters.start_lr,
                "weight_decay": simulation_parameters.weight_decay,
                "step_size": simulation_parameters.step_size,
                "gamma": simulation_parameters.gamma,
                "tau": model_config.tau,
                "N": system_model_params.N,
                "M": system_model_params.M,
                "T": system_model_params.T,
            }
        )
        # Perform simulation training and evaluation stages
        print("Training Model...")
        model, loss_train_list, loss_valid_list = train(
            training_parameters=simulation_parameters,
            model_name=simulation_filename,
            saving_path=saving_path,
        )
        # Save model weights
        if commands["SAVE_MODEL"]:
            torch.save(
                model.state_dict(),
                saving_path / "final_models" / Path(simulation_filename),
            )
        # Plots saving
        if commands["SAVE_TO_FILE"]:
            plt.savefig(
                simulations_path
                / "results"
                / "plots"
                / Path(dt_string_for_save + r".png")
            )
        else:
            plt.show()

    # Evaluation stage
    if commands["EVALUATE_MODE"]:
        # Initialize figures dict for plotting
        figures = initialize_figures()
        # Define loss measure for evaluation
        if model_config.diff_method == "mvdr":
            criterion, subspace_criterion = set_criterions("sisnr")
        if model_config.diff_method == "music":
            criterion, subspace_criterion = set_criterions("Spectrum_Loss")
        # Load datasets for evaluation
        # if not (commands["CREATE_DATA"] or commands["LOAD_DATA"]):
        #     test_dataset, generic_test_dataset, samples_model = load_datasets(
        #         system_model_params=system_model_params,
        #         model_type=model_config.model_type,
        #         samples_size=samples_size,
        #         datasets_path=datasets_path,
        #         train_test_ratio=train_test_ratio,
        #     )

        # Generate DataLoader objects
        model_test_dataset = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, drop_last=False
        )
        # Load pre-trained model
        if not commands["TRAIN_MODEL"]:
            # Define an evaluation parameters instance
            simulation_parameters = (
                TrainingParams()
                .set_model(model=model_config)
                .load_model(
                    loading_path=saving_path
                    / checkpoint
                )
            )
            model = simulation_parameters.model
        # print simulation summary details
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            phase="evaluation",
            parameters=simulation_parameters,
        )
        wandb_name = checkpoint + "_test"
        # Evaluate DNN models, augmented and subspace methods
        test_dnn_model(
            model=model,
            dataset=model_test_dataset,
            criterion=criterion,
            plot_spec=False,
            figures=figures,
            model_type=model_config.model_type,
            wandb_name = wandb_name,
        )
    plt.show()
    print("end")

#runai-cmd --name test -g 1 --cpu-limit 32 -- "conda activate SubspaceNetEnv && python /gpfs0/bgu-br/users/tatarjit/model-based-nir/main.py"
