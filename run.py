import argparse
import random

import pandas as pd
import numpy as np
import yaml
import os
import gc
import time

from models import models
from utils.data import split_into_samples, data_generator, get_files_in_dir
from evaluation.metrics import apply_metrics
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested


def main():
    
    start_time_job = time.time()
    parser = argparse.ArgumentParser(description="Runner for ML model training for Short-Term Goal Prediction.",
                                     usage="Run within conda environment using 'python3 run.py CONFIG' and "
                                           "add the necessary command line arguments")
    parser.add_argument("config",
                        metavar='CONFIGURATION',
                        help='name of the configuration file in configs/', type=str)
    
    parser.add_argument("--lookback", type=int)
    parser.add_argument("--outputpath", type=str)
    parser.add_argument("--pred_goal", type=str)
    parser.add_argument("--outlook", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--layerCnt", type=int)
    parser.add_argument("--piselection", type=str)    
    parser.add_argument("--piconfiglist", type=str, required=False)

    parser.add_argument('--seed', dest="random_seed", metavar="SEED",
                        help="optional seed to make random stuff deterministic", type=int, required=False)

    args = parser.parse_args()

    random_seed = args.random_seed if args.random_seed is not None else random.randint(0, 200)

    with open(os.path.join("configs", args.config), "r") as stream:
        config = yaml.safe_load(stream)

    print("Argument parsing successful.")

    results = pd.DataFrame()
    mode = config.get("mode", "classification")
    lookback = args.lookback
    outlook = args.outlook
    MLmodel = args.model
    PIselection = args.piselection
    
    print(f"arguments:{args.layerCnt}")
    ChannelCnt = args.layerCnt
    OutputPath = args.outputpath
    PredGoal = args.pred_goal

    config.get("sampling_config")["window_length_lookback"] = lookback
    config.get("sampling_config")["window_length_outlook"] = outlook
    config.get("sampling_config")["training_goal"] = PredGoal
    config["output_dir_path"] = OutputPath
                
    cover = config.get("sampling_config").get("hidden")

    print(lookback)
    print(outlook)
    print(cover)
    print(PIselection)    

    if os.path.isdir(config.get("dataset_path")):
        x_train, y_train, labels_train, x_val, y_val, labels_val, x_test, y_test, labels_test = split_into_samples(
            data_generator(get_files_in_dir(config.get("dataset_path"))), **config.get("sampling_config")
        )
    else:
        x_train, y_train, labels_train, x_val, y_val, labels_val, x_test, y_test, labels_test = split_into_samples(
            data_generator([config.get("dataset_path")]), **config.get("sampling_config")
        )

    print("Dataset sampling successful.")
    print(f"Ratio of positive sample in data set: {y_train.sum() / len(y_train)} (train), {y_val.sum() / len(y_val)} (val), {y_test.sum() / len(y_test)} (test)")


    if args.piconfiglist is not None:                
            print(f"Read {args.piconfiglist}")
            # Lese den Inhalt der Textdatei isCorner_bestPIs_Combinations_1.txt
            with open(args.piconfiglist, "r") as file:
                combinations_content = file.read().splitlines()
            # Füge den Inhalt der Textdatei zur pi_config_list hinzu
            config["pi_config_list"] = [comb.split(', ') for comb in combinations_content]
            print("Successfully readed")
 
    pi_list = config.get("sampling_config").get("pi_list")
    pi_config_list = config.get("pi_config_list")
    total_iterations = len(pi_config_list)
    iteration_count = 0

    for model_name in config.get("models").keys():
        # TODO: Maybe check whether model is compatible with data etc.
        start_time_model = time.time()
        iteration_count = 0
 
        print(model_name)
                
        for iteration, pi in enumerate(pi_config_list, start=1):            
            iteration_count += 1
            print(f"Iteration {iteration_count}/{total_iterations}")
            start_time = time.time()
            
            if model_name == "NNClassifier" :
                config.get("models").get(model_name).get("model_config").get("model_config")["time_steps"] = lookback
                config.get("models").get(model_name).get("model_config").get("model_config")["num_channels"] = ChannelCnt
                    
            if isinstance(pi, str):
                # List of PIs (e.g. ["Dangerousity", "Dangerousity_dif"])
                pi = [pi]

            pi_indices = [pi_list.index(p) for p in pi]
            pi_string = '-'.join(pi)

            if models.get(model_name).input_format() == "time":
                x_train_model = x_train[:, :, pi_indices]
                x_val_model = x_val[:, :, pi_indices]
                x_test_model = x_test[:, :, pi_indices]

            elif models.get(model_name).input_format() == "time_series":
                x_train_model = np.transpose(x_train[:, :, pi_indices], (0, 2, 1))
                x_val_model = np.transpose(x_val[:, :, pi_indices], (0, 2, 1))
                x_test_model = np.transpose(x_test[:, :, pi_indices], (0, 2, 1))

                x_train_model = from_3d_numpy_to_nested(x_train_model)
                x_val_model = from_3d_numpy_to_nested(x_val_model)
                x_test_model = from_3d_numpy_to_nested(x_test_model)

            else:
                x_train_model = np.reshape(x_train[:, :, pi_indices], (len(x_train), -1))
                x_val_model = np.reshape(x_val[:, :, pi_indices], (len(x_val), -1)) if len(x_val) != 0 else x_val
                x_test_model = np.reshape(x_test[:, :, pi_indices], (len(x_test), -1)) if len(x_test) != 0 else x_test

            model = models.get(model_name)(config.get("models").get(model_name).get("model_config"), random_seed)
                       
            model.train(x_train_model, y_train, x_val_model, y_val)

            if len(y_test) != 0:
                y_test_pred = model.predict(x_test_model)
           
            if config.get("save_models"):
                output_path = os.path.join(config.get("output_dir_path"), "models", f"{model_name}-{pi_string}")
                model.save(output_path)
            
            del model
            gc.collect()

            if len(y_test) != 0:
                res = apply_metrics(y_test, y_test_pred, mode)
                res_df = pd.DataFrame([res])
                res_entry = pd.DataFrame([[model_name, pi_string, lookback, outlook, cover]],columns=['Model', 'PI', 'Lookback', 'Outlook', 'Cover'])
                res_entry = pd.concat([res_entry, res_df], axis=1)
                results = pd.concat([results, res_entry])
            
            print(f"Computation for {model_name} and PI(s) {pi_string} was successful.")
            
            ##Elapsed time
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Aufteilung in Stunden, Minuten und Sekunden
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            # Formatierung in lesbares Format
            elapsed_time_formatted = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
            
            print(f"Elapsed time: {elapsed_time_formatted}")
        print(f"Computation for {model_name} and all PI(s) was successful.")
            
        ##Elapsed time
        end_time_model = time.time()
        elapsed_time = end_time_model - start_time_model
        
        # Aufteilung in Stunden, Minuten und Sekunden
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Formatierung in lesbares Format
        elapsed_time_formatted = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        
        print(f"Elapsed time for complete model : {elapsed_time_formatted}")
                
    
    for i in range(100):
        if i == 0:
            output_path = os.path.join(config.get("output_dir_path"),f"{mode}-{lookback}_{outlook}_{MLmodel}_{PIselection}_{PredGoal}.csv")
        else:
            output_path = os.path.join(config.get("output_dir_path"),f"{mode}-{lookback}_{outlook}_{MLmodel}_{PIselection}_{PredGoal}-{i}.csv")
        if not os.path.exists(output_path):
            break
    results.to_csv(output_path, index=False, sep=";", decimal=",")
    print(f"DONE! Wrote new CSV file in {output_path}")

    ##Elapsed time
    end_time_job = time.time()
    elapsed_time = end_time_job - start_time_job
    
    # Aufteilung in Stunden, Minuten und Sekunden
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Formatierung in lesbares Format
    elapsed_time_formatted = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    
    print(f"Elapsed time for complete Job : {elapsed_time_formatted}")

if __name__ == "__main__":
    main()
