import numpy as np
from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
import math
import os
from typing import List, Iterable
from multiprocessing import Pool, cpu_count
from itertools import chain


def get_files_in_dir(path: str, prefix: str = None, suffix: str = None) -> List[str]:
    """
    List all files in directory with optional prefix or suffix

    :param path: Path of directory that should be searched in
    :param prefix: Optional prefix of all valid files
    :param suffix: Optional suffix of all valid files
    :return: List of all (valid) file paths in that folder
    """
    return [os.path.join(path, file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))
            and (prefix is None or file.startswith(prefix))
            and (suffix is None or file.endswith(suffix))]


def data_generator(filepaths: List[str]) -> Iterable[DataFrame]:
    """
    Optimize the runtime of the data loading by using this function as a generator that can be used for the splits

    :param filepaths: List of filepaths of the single files (as generated from get_files_in_dir)
    :return: Iterable of pandas dataframes (one for each match)
    """
    for file in filepaths:
        yield read_file(file)


def get_all_matches_in_dataframe(matches: Iterable[DataFrame]):
    return concat(matches, ignore_index=True)


def split_into_samples(matches: Iterable[DataFrame],
                       window_length_lookback: int,
                       window_length_outlook: int,
                       pi_list: list,
                       training_goal: str = "isGoal",
                       folding=("any_above", 0),
                       hidden: int = 0,
                       sample_rate: int = 1,
                       split: List[float] = None,
                       split_in_matches: bool = False,
                       normalize: bool = True,
                       parallelize: bool = True):
    """
    Generate samples from matches, and optionally split into train(-val)-test and normalize the datasets using
    min-max-scaling. Method uses a rolling window approach to generate samples

    :param matches: Iterable with match dataframes
    :param window_length_lookback: Number of frames that are used for one training sample sequence
    :param window_length_outlook: Number of frames to look into future to see if there is a goal
    :param pi_list: List indicating the names of performance indicators that should be used
    :param training_goal: Prediction goal (e.g. goals)
    :param folding: Folding of training goal: ("any_above", value) or "sum" or "mean"
    :param hidden: Interval to hide between lookback and outlook windows
    :param sample_rate: Step size of rolling window sampling
    :param split: List with preferred split, can have 2 (train-test) or 3 (train-val-test) entries
    :param split_in_matches: whether to split each match into train(-val)-test samples (produces fewer samples)
    :param normalize: whether to normalize the data with min-max scaling (fitted to train set)
    :param parallelize: whether to parallelize the generation of samples over all matches
    :return: (x,y,labels) * [1,2 or 3] depending on which split is selected
    """
    x_train, x_val, x_test, y_train, y_val, y_test, labels_train, labels_val, labels_test = [], [], [], [], [], [], [], [], []

    # Check whether given split argument is valid
    if split is not None:
        assert len(split) in [2, 3], f"Argument split (list) has to be of length 2 or 3 but has length {len(split)}."
        if len(split) == 2:
            split = [split[0]] + [0.0] + [split[1]]
        split = np.array(split)
        assert split.sum() == 1, f"Sum of train-, val- and test-fraction has to be 1 but is {split.sum()}."

    # Generate samples with rolling window approach from all matches

    if parallelize:
        with Pool(cpu_count()) as p:
            ids = []
            def collect_results(results):
                for i, res in enumerate(results):
                    if not isinstance(res, list) or len(res) > 0:
                        if i == 0:
                            x_train.append(res)
                        elif i == 1:
                            y_train.append(res)
                        elif i == 2:
                            labels_train.append(res)
                        elif i == 3:
                            x_val.append(res)
                        elif i == 4:
                            y_val.append(res)
                        elif i == 5:
                            labels_val.append(res)
                        elif i == 6:
                            x_test.append(res)
                        elif i == 7:
                            y_test.append(res)
                        elif i == 8:
                            labels_test.append(res)
                        else:
                            ids.append(res)

            for i, match in enumerate(matches):
                p.apply_async(process_single_match,
                              args=(match, split, split_in_matches, window_length_lookback, hidden,
                                    window_length_outlook, pi_list, training_goal, folding, sample_rate, i),
                              callback=collect_results)
            p.close()
            p.join()
            sort_indices = np.argsort(np.array(ids))
            x_train = [x_train[i] for i in sort_indices]
            x_val = [x_val[i] for i in sort_indices]
            x_test = [x_test[i] for i in sort_indices]
            y_train = [y_train[i] for i in sort_indices]
            y_val = [y_val[i] for i in sort_indices]
            y_test = [y_test[i] for i in sort_indices]
            labels_train = [labels_train[i] for i in sort_indices]
            labels_val = [labels_val[i] for i in sort_indices]
            labels_test = [labels_test[i] for i in sort_indices]
    else:
        for match in matches:
            # Split each match into train-val-test
            if split is not None and split_in_matches:
                match_len = len(match)
                split_ids = np.around(np.cumsum(split)[:2] * match_len).astype(int)
                data_train = match[:split_ids[0]]
                data_val = match[split_ids[0]:split_ids[1]]
                data_test = match[split_ids[1]:]

                if len(data_train) >= window_length_lookback + hidden + window_length_outlook:
                    x_train_match, y_train_match, labels_train_match = split_match_into_samples(
                        data_train, window_length_lookback, window_length_outlook, pi_list, training_goal, folding,
                        hidden, sample_rate)
                    x_train.append(x_train_match)
                    y_train.append(y_train_match)
                    labels_train += labels_train_match

                if len(data_val) >= window_length_lookback + hidden + window_length_outlook:
                    x_val_match, y_val_match, labels_val_match = split_match_into_samples(
                        data_val, window_length_lookback, window_length_outlook, pi_list, training_goal, folding,
                        hidden, sample_rate)
                    x_val.append(x_val_match)
                    y_val.append(y_val_match)
                    labels_val += labels_val_match

                if len(data_test) >= window_length_lookback + hidden + window_length_outlook:
                    x_test_match, y_test_match, labels_test_match = split_match_into_samples(
                        data_test, window_length_lookback, window_length_outlook, pi_list, training_goal, folding,
                        hidden, sample_rate)
                    x_test.append(x_test_match)
                    y_test.append(y_test_match)
                    labels_test += labels_test_match
            # Split into train-val-test afterwards
            else:
                x_match, y_match, labels_match = split_match_into_samples(
                    match, window_length_lookback, window_length_outlook, pi_list, training_goal, folding, hidden,
                    sample_rate)
                x_train.append(x_match)
                y_train.append(y_match)
                labels_train += labels_match

    # Transform lists to numpy arrays
    x_train = np.concatenate(x_train) if len(x_train) > 0 else np.array([])
    y_train = np.concatenate(y_train) if len(y_train) > 0 else np.array([])
    x_val = np.concatenate(x_val) if len(x_val) > 0 else np.array([])
    y_val = np.concatenate(y_val) if len(y_val) > 0 else np.array([])
    x_test = np.concatenate(x_test) if len(x_test) > 0 else np.array([])
    y_test = np.concatenate(y_test) if len(y_test) > 0 else np.array([])
    labels_train = list(chain.from_iterable(labels_train))
    labels_val = list(chain.from_iterable(labels_val))
    labels_test = list(chain.from_iterable(labels_test))

    # Split into train-val-test in case split is not in matches
    if split is not None and not split_in_matches:
        num_samples = len(x_train)
        split_ids = np.around(np.cumsum(split)[:2] * num_samples).astype(int)
        x_val = x_train[split_ids[0] + window_length_lookback // 2: split_ids[1] - window_length_lookback // 2]
        x_test = x_train[split_ids[1] + window_length_lookback // 2:]
        x_train = x_train[:split_ids[0] - window_length_lookback // 2]

        y_val = y_train[split_ids[0] + window_length_lookback // 2: split_ids[1] - window_length_lookback // 2]
        y_test = y_train[split_ids[1] + window_length_lookback // 2:]
        y_train = y_train[:split_ids[0] - window_length_lookback // 2]

        labels_val = labels_train[split_ids[0] + window_length_lookback // 2: split_ids[1] - window_length_lookback // 2]
        labels_test = labels_train[split_ids[1] + window_length_lookback // 2:]
        labels_train = labels_train[:split_ids[0] - window_length_lookback // 2]

    # Normalize ALL variables in x by min-max scaling
    if normalize:
        scaler = MinMaxScaler()
        if len(x_train) > 0:
            x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
        if len(x_test) > 0:
            x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
        if len(x_val) > 0:
            x_val = scaler.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(x_val.shape)

    if split is None:
        return x_train, y_train, labels_train

    if split[1] == 0:
        return x_train, y_train, labels_train, x_test, y_test, labels_test
    else:
        return x_train, y_train, labels_train, x_val, y_val, labels_val, x_test, y_test, labels_test


def process_single_match(match, split, split_in_matches, window_length_lookback, hidden, window_length_outlook, pi_list,
                         training_goal, folding, sample_rate, p_id):
    # Split each match into train-val-test
    if split is not None and split_in_matches:
        match_len = len(match)
        split_ids = np.around(np.cumsum(split)[:2] * match_len).astype(int)
        data_train = match[:split_ids[0]]
        data_val = match[split_ids[0]:split_ids[1]]
        data_test = match[split_ids[1]:]

        x_train_match, y_train_match, labels_train_match, x_val_match, y_val_match, labels_val_match, \
        x_test_match, y_test_match, labels_test_match = [], [], [], [], [], [], [], [], []

        if len(data_train) >= window_length_lookback + hidden + window_length_outlook:
            x_train_match, y_train_match, labels_train_match = split_match_into_samples(
                data_train, window_length_lookback, window_length_outlook, pi_list, training_goal, folding, hidden,
                sample_rate)

        if len(data_val) >= window_length_lookback + hidden + window_length_outlook:
            x_val_match, y_val_match, labels_val_match = split_match_into_samples(
                data_val, window_length_lookback, window_length_outlook, pi_list, training_goal, folding, hidden,
                sample_rate)

        if len(data_test) >= window_length_lookback + hidden + window_length_outlook:
            x_test_match, y_test_match, labels_test_match = split_match_into_samples(
                data_test, window_length_lookback, window_length_outlook, pi_list, training_goal, folding, hidden,
                sample_rate)

        return x_train_match, y_train_match, labels_train_match, x_val_match, y_val_match, labels_val_match, \
               x_test_match, y_test_match, labels_test_match, p_id
    # Split into train-val-test afterwards
    else:
        x_match, y_match, labels_match = split_match_into_samples(
            match, window_length_lookback, window_length_outlook, pi_list, training_goal, folding, hidden, sample_rate)
        return x_match, y_match, labels_match, [], [], [], [], [], [], p_id


def read_file(path: str) -> DataFrame:
    """
    Returns Pandas data frame for a given csv file

    :param path: path to csv file
    :return: unmodified dataframe
    """
    return read_csv(path, delimiter=";", decimal=",")


def split_match_into_samples(data: DataFrame,
                             window_length_lookback: int,
                             window_length_outlook: int,
                             pi_list: list,
                             training_goal: str = "isGoal",
                             folding=("any_above", 0),
                             hidden: int = 0,
                             sample_rate: int = 1):
    """
    Splits a given pandas dataframe into training samples of length num_steps for distinct games

    :param data: pandas dataframe
    :param window_length_lookback: number of frames that are used for one training sample sequence
    :param window_length_outlook: number of frames to look into future to see if there is a goal
    :param pi_list: list indicating the names of performance indicators that should be used
    :param training_goal: prediction goal (e.g. goals)
    :param folding: folding of training goal: ("any_above", value) or "sum" or "mean"
    :param hidden: interval to hide between lookback and outlook windows
    :param sample_rate: rate in which samples should be sampled (step size of rolling window)
    :return: (x, y, labels) -> x: training samples (sequence of selected PI values), y: training labels,
    labels: list of corresponding labels
    """
    # Dimensions of x: num_samples, num_pis, window_length_lookback
    # Dimensions of y: num_samples,

    # Currently |-----------[---lookback---|---hidden---|---outlook---]----|
    # with only taking intervals that are fully covered in the game.
    # Currently doesn't care if there is a goal in the lookback window

    label_list = []
    x_list = []
    y_list = []
    for i, frames in enumerate(data.rolling(window_length_lookback + hidden + window_length_outlook)):
        if i % sample_rate != 0:
            continue
        index_first = frames.first_valid_index()
        if len(frames) == window_length_lookback + hidden + window_length_outlook \
                and not frames.isnull().values.any() \
                and (frames["MatchID"][index_first] == frames["MatchID"]).all() \
                and (frames["Team"][index_first] == frames["Team"]).all() \
                and (frames["Opponent"][index_first] == frames["Opponent"]).all() \
                and frames["Intervall"].is_monotonic_increasing and frames["Intervall"].is_unique:

            assert training_goal in frames, "The given prediction target can not be found in the dataset."

            x = frames[pi_list].iloc[:window_length_lookback].to_numpy().T
            x_list.append(x)

            if folding == "sum":
                y = frames[training_goal].iloc[window_length_lookback + hidden:].sum()
            elif folding == "mean":
                y = frames[training_goal].iloc[window_length_lookback + hidden:].mean()
            else:
                fold, fold_arg = folding
                if fold == "any_above":
                    y = (frames[training_goal].iloc[window_length_lookback + hidden:] > fold_arg).any()
                elif fold == "any_below":
                    y = (frames[training_goal].iloc[window_length_lookback + hidden:] < fold_arg).any()
                else:
                    raise ValueError("Folding operator for prediction target is not supported.")
            y_list.append(y)
            label_list.append(
                f"{frames['MatchID'][index_first]}-{frames['Team'][index_first]}-{frames['Intervall'][index_first]}")
    x = np.transpose(np.array(x_list), (0, 2, 1))
    y = np.array(y_list)
    return x, y, label_list
