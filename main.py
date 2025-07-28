# Imports
import sys
import torch
import os
import matplotlib.pyplot as plt
import warnings
from src.system_model import SystemModelParams
from src.data_handler import *
from src.criterions import set_criterions
from src.training import *
from src.evaluation import test_dnn_model
from pathlib import Path
from src.models import ModelGenerator
import wandb
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
wandb.login(key="d55bad0e0b8a03b9bdfa4beeacf508cd29e1e398")

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
    saving_path = external_data_path / "weights_cluster"

    # create folders if not exists
    saving_path.mkdir(parents=True, exist_ok=True)

    # Operations commands
    train_mode = False
    commands = {
        "LOAD_RECORDINGS": True,  # Load recordings from external file
        "LOAD_MODEL": False,  # Load specific model for training
        "TRAIN_MODEL": train_mode,  # Applying training operation
        "SAVE_MODEL": True,  # Saving tuned model
        "EVALUATE_MODE": not train_mode,  # Evaluating desired algorithms
    }
    # Define system model parameters
    system_model_params = (
        SystemModelParams()
        .set_parameter("N", 15)
        .set_parameter("M", 1)
        .set_parameter("T", 157)
        .set_parameter("signal_type", "NarrowBand")
        .set_parameter("signal_nature", "non-coherent")
    )
    # Generate model configuration
    model_config = (
        ModelGenerator()
        .set_model_type("SubspaceNet")
        .set_diff_method("music") # mvdr / music
        .set_tau(8)
        .set_model(system_model_params)
    )
    # Print new simulation intro
    print("------------------------------------")
    print("---------- New Simulation ----------")
    print("------------------------------------")
    print("date and time =", dt_string)
    # Load recordings from external file
    set_unified_seed()

    if commands["LOAD_RECORDINGS"]:
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
            .set_schedular(step_size=10, gamma=0.3, total_epochs=300, start_lr=1e-3, warmup_epochs=1)
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

    # Evaluation stage
    if commands["EVALUATE_MODE"]:
        # Define loss measure for evaluation
        if model_config.diff_method == "mvdr":
            criterion, subspace_criterion = set_criterions("sisnr")
        if model_config.diff_method == "music":
            criterion, subspace_criterion = set_criterions("Spectrum_Loss")

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
        test_dnn_model(
            model=model,
            dataset=model_test_dataset,
            criterion=criterion,
            plot_spec=False,
            model_type=model_config.model_type,
            wandb_name = wandb_name,
        )
    plt.show()
    print("end")

#runai-cmd --name test -g 1 --cpu-limit 32 -- "conda activate SubspaceNetEnv && python /gpfs0/bgu-br/users/tatarjit/model-based-nir/main.py"
