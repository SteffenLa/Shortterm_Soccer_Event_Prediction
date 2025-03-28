import math
from typing import Optional, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.signal import savgol_filter
import pandas as pd
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested

from utils.data import split_match_into_samples


def plot_match(df: pd.DataFrame, match_id: str, team_id: str, pi: str, unit: Tuple[str, float] = ("Minutes", 1)):
    """
    Visualization of the evolution of a PI over a single match. Shows team, opponent and difference, as well as the
    moments where goals happen.

    :param df: Dataframe with all matches
    :param match_id: e.g. "DFL-MAT-00278A"
    :param team_id: e.g. "DFL-CLU-00000J"
    :param pi: e.g. "Dangerousity" - do not use ".._dif" here, it is automatically used in plot
    :param unit: Unit of one time step, e.g. ("Minutes", 1)
    :return: Figure with plot
    """
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(20, 4))
    match_part = df[(df["MatchID"] == match_id).__and__(df["Team"] == team_id)]
    x_axis = [unit[1] * j for j in range(len(match_part))]
    sns.lineplot(ax=ax, x=x_axis, y=match_part[pi], color="blue", label="Home")
    sns.lineplot(ax=ax, x=x_axis,
                 y=match_part[f"{pi}_dif"] - match_part[pi],
                 label="Away",
                 color="red")
    sns.lineplot(ax=ax, x=x_axis, y=match_part[f"{pi}_dif"], color="gray", label="Difference")
    ax.set_ylim([- match_part[pi].abs().max(), match_part[pi].abs().max()])
    ax.legend()
    fig.suptitle("PI values over one match")
    ax.set_ylabel(pi)
    ax.set_xlabel(f"Time steps (in {unit[0]})")
    for i, row in match_part.iterrows():
        if row["isGoal_dif"] > 0:
            ax.axvline(x=i - match_part.first_valid_index(), color="darkblue", ls="--")
        elif row["isGoal_dif"] < 0:
            ax.axvline(x=i - match_part.first_valid_index(), color="darkred", ls="--")
    fig.tight_layout()
    return fig


def plot_events(df: pd.DataFrame, pi: str, num_timesteps: int, num_samples: Optional[int] = None,
                cover: int = 0, event: Tuple[str, int] = ("isGoal", 0), unit: Tuple[str, float] = ("Minutes", 1),
                box_plot: bool = False, box_plot_whis=(5, 95)):
    """
    Plot the evolution of a PI before a certain event in the data (e.g. before a goal)

    :param df: Dataframe with all matches
    :param pi: e.g. "Dangerousity_dif"
    :param num_timesteps: Number of timesteps to look into past from the event
    :param num_samples: Number of samples to visualize, if None: use all available
    :param cover: Number of timesteps to cover before the event happens
    :param event: Which kind of event to look at: e.g. whether a goal happens (whether "isGoal" is > 0)
    :param unit: Unit of one time step, e.g. ("Minutes", 1)
    :param box_plot: Whether to add a box plot to first plot
    :param box_plot_whis: What confidence interval to use for the box plot
    :return: Figure with plot
    """
    sns.set_theme(style="whitegrid")

    fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 10))
    fig.suptitle(f'Analysis of {pi} before {event[0]} > {event[1]}')

    goal_indices = df.index[df[event[0]] > event[1]]
    num_goals_total = len(goal_indices)
    value_list = None
    x_axis = [unit[1] * j for j in range(-num_timesteps + 1 - cover, 1 - cover, 1)]
    sample_counter = 0
    if num_samples is None:
        num_samples = num_goals_total
    for i in range(num_samples):
        if num_samples == num_goals_total:
            index = goal_indices[i]
        else:
            index = random.choice(goal_indices)
        if index < num_timesteps:
            # print(f"Index: {index} < {num_timesteps}")
            continue
        data = df.iloc[index - num_timesteps + 1 - cover:index + 1 - cover]
        if data["MatchID"].nunique() != 1 or data["Team"].nunique() != 1:
            # print(f"Unique: {data['MatchID'].nunique()}, Team: {data['Team'].nunique()}")
            continue
        sample_counter += 1
        pi_values = data[pi].tolist()
        if value_list is None:
            value_list = np.array(pi_values)
        else:
            value_list = np.vstack([value_list, np.array(pi_values)])
        if i < 300:
            sns.lineplot(ax=ax1, x=x_axis, y=pi_values)

    minor_ticks = np.arange(-num_timesteps + 1 - cover, 0, 1) * unit[1]
    ax1.set_xticks(minor_ticks, minor=True)
    ax1.grid(which='minor', alpha=0.2)

    ax1.set_title("All Samples")
    ax1.set_ylabel(pi)
    ax1.set_xlabel("Time steps before event (in " + unit[0] + ")")

    mean_values = np.mean(value_list, 0)
    std_values = np.std(mean_values, 0)

    mean_denoised = savgol_filter(mean_values, num_timesteps // 4, 5)
    std_lower = mean_values - std_values
    std_upper = mean_values + std_values

    sns.lineplot(ax=ax2, x=x_axis, y=mean_values, label="Mean values")
    sns.lineplot(ax=ax2, x=x_axis, y=mean_denoised, label="Smoothed trend")
    ax2.fill_between(x_axis, std_lower, std_upper, alpha=.3)

    ax2.set_xticks(minor_ticks, minor=True)
    ax2.grid(which='minor', alpha=0.2)

    ax2.set_title("Mean values of samples")
    ax2.set_ylabel(pi)
    ax2.set_xlabel("Time steps before event (in " + unit[0] + ")")

    if box_plot is not None:
        if box_plot == 1:
            ax3 = ax1.twinx()
        else:
            ax3 = ax2.twinx()
        steps_per_minute = int(1 / unit[1])
        minute_indices = list(range(-steps_per_minute + cover - 1, -num_timesteps, -steps_per_minute))
        minute_list = value_list[:, minute_indices]
        minute_x_axis = np.array(list(range(min(math.floor(-unit[1] * cover), 0), math.floor(min(x_axis)), -1)))
        if box_plot == 1:
            ax3.set_ylim(ax1.get_ylim())
        else:
            ax3.set_ylim(ax2.get_ylim())
        for i in range(len(minute_x_axis)):
            bplot = ax3.boxplot(minute_list[:, i], positions=[minute_x_axis[i]], widths=0.25, whis=box_plot_whis,
                                patch_artist=True)
            for patch in bplot['boxes']:
                patch.set_facecolor('gray')
                patch.set_alpha(0.8)
            for median in bplot['medians']:
                median.set_color('red')
    fig.tight_layout()

    print(f"Number of actually used samples: {sample_counter} of {num_goals_total}")
    return fig


def plot_all_matches(df, pi, unit: Tuple[str, float] = ("Minutes", 1)):
    """
    Plot the evolution of a PI over all matches and the distribution of goals over all matches

    :param df: Dataframe
    :param pi: e.g. "Dangerousity_dif"
    :param unit: Unit of one time step, e.g. ("Minutes", 1)
    :return: Figure with plot
    """
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(20, 12))
    max_length = -1
    num_samples = 0
    goal_indices = []
    values = []

    for match_id in df["MatchID"].unique():
        match_part = df[df["MatchID"] == match_id]
        for team_id in match_part["Team"].unique():
            match_team_part = match_part[match_part["Team"] == team_id]
            x_axis = [unit[1] * j for j in range(len(match_team_part))]
            sns.lineplot(ax=ax1, x=x_axis, y=match_team_part[pi], ci=None)
            if len(match_team_part) > max_length:
                max_length = len(match_team_part)
            num_samples += 1
            values.append(match_team_part[pi])
            for i, row in match_team_part.iterrows():
                if row["isGoal_dif"] > 0:
                    goal_indices.append(i - match_team_part.first_valid_index())

    fig.suptitle("PI values over all matches")
    ax1.set_title("All Samples")
    ax1.set_ylabel(pi)
    ax1.set_xlabel("Time steps (in " + unit[0] + ")")

    print("First plot finished")

    all_values = np.full((num_samples, max_length), np.nan)
    for i, value in enumerate(values):
        all_values[i, :len(value)] = value

    mean_values = np.nanmean(all_values, 0)
    std_values = np.nanstd(all_values, 0)
    x_axis = [unit[1] * j for j in range(max_length)]
    mean_denoised = savgol_filter(mean_values, max_length // 5, 5)

    std_lower = mean_values - std_values
    std_upper = mean_values + std_values

    sns.lineplot(ax=ax2, x=x_axis, y=mean_values, label="Mean values")
    sns.lineplot(ax=ax2, x=x_axis, y=mean_denoised, label="Smoothed trend")
    ax2.fill_between(x_axis, std_lower, std_upper, alpha=.3)

    ax2.set_title("Mean values of PIs")
    ax2.set_ylabel(pi)
    ax2.set_xlabel("Time steps (in " + unit[0] + ")")
    # ax2.set_ylim([mean_values.min() - std_values.max() // 1.2, mean_values.max() + std_values.max() // 1.2])

    print("Second plot finished")

    sns.histplot(ax=ax3, x=goal_indices, kde=False, binwidth=max(1, max_length // 100), legend=False)
    ax3.set_title("Goal distribution over all matches")
    ax3.set_ylabel("Goals")
    ax3.set_xlabel("Time steps")
    ax3.set_xlim(0, max_length)

    goal_distribution = np.zeros(max_length)
    for index in goal_indices:
        goal_distribution[index] += 1
    num_samples_timesteps = np.sum(~np.isnan(all_values), 0)
    ratio_per_timestep = goal_distribution / num_samples_timesteps
    mean_ratio_denoised = savgol_filter(ratio_per_timestep, max_length // 5, 5)

    ax4 = ax3.twinx()
    x_axis = np.array(x_axis)
    sns.lineplot(ax=ax4, x=x_axis + 0.5, y=ratio_per_timestep, legend=False, label="Ratio per time step", ls="--",
                 color="darkgreen")
    sns.lineplot(ax=ax4, x=x_axis + 0.5, y=mean_ratio_denoised, legend=False, label="Smoothed Ratio", ls="--",
                 color="darkblue")
    ax4.legend()
    print("Third plot finished")

    fig.tight_layout()

    return fig


def plot_predictions(df, clf, match_id, value,
                     window_length_lookback: int,
                     window_length_outlook: int,
                     pi_list: list,
                     training_goal: str = "isGoal",
                     folding=("any_above", 0),
                     hidden: int = 0,
                     **kwargs):
    match = df[df["MatchID"] == match_id]
    match_part_home = match[match["Team"] == match["Team"].unique()[0]]
    match_part_away = match[match["Team"] == match["Team"].unique()[1]]
    x_home, y_home, label_home = split_match_into_samples(match_part_home, window_length_lookback,
                                                          window_length_outlook, pi_list,
                                                          training_goal, folding, hidden, sample_rate=1)
    x_away, y_away, label_away = split_match_into_samples(match_part_away, window_length_lookback,
                                                          window_length_outlook, pi_list,
                                                          training_goal, folding, hidden, sample_rate=1)

    if clf.input_format() == "time_series":
        x_home = np.transpose(x_home, (0, 2, 1))
        x_home = from_3d_numpy_to_nested(x_home)
        x_away = np.transpose(x_away, (0, 2, 1))
        x_away = from_3d_numpy_to_nested(x_away)
    else:
        x_home = np.reshape(x_home, (len(x_home), -1))
        x_away = np.reshape(x_away, (len(x_away), -1))

    y_pred_home = clf.predict(x_home)
    y_pred_away = clf.predict(x_away)

    y_proba_home = clf.predict_proba(x_home)
    y_proba_away = clf.predict_proba(x_away)

    if y_proba_home.ndim == 2:
        y_proba_home = y_proba_home[:, 0]
        y_proba_away = y_proba_away[:, 0]

    match_len = len(match_part_home)

    if value == "pred":
        y_running_home = compute_running_mean(y_pred_home, window_length_lookback, window_length_outlook, hidden)
        y_running_away = compute_running_mean(y_pred_away, window_length_lookback, window_length_outlook, hidden)
    elif value == "proba":
        y_running_home = compute_running_mean(y_proba_home, window_length_lookback, window_length_outlook, hidden)
        y_running_away = compute_running_mean(y_proba_away, window_length_lookback, window_length_outlook, hidden)
    else:
        raise ValueError(f"Method only accepts 'pred' and 'proba' for value parameter but got {value}")

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(20, 4))
    x_axis = list(range(match_len))
    sns.lineplot(ax=ax, x=x_axis, y=match_part_home[pi_list[0]], color="cornflowerblue", label="Home")
    sns.lineplot(ax=ax, x=x_axis,
                 y=-match_part_away[pi_list[0]],
                 label="Away",
                 color="lightcoral")
    sns.lineplot(ax=ax, x=x_axis, y=match_part_home[f"{pi_list[0]}_dif"], color="silver", label="Difference")
    ax.set_ylim([- match_part_away[pi_list[0]].max(), match_part_away[pi_list[0]].max()])
    ax.legend()
    fig.suptitle("PI values over one match")
    ax.set_ylabel(pi_list[0])
    ax.set_xlabel("Time steps")
    for i, row in match_part_home.iterrows():
        if row["isGoal_dif"] > 0:
            ax.axvline(x=i - match_part_home.first_valid_index(), color="darkblue", ls="--")
        elif row["isGoal_dif"] < 0:
            ax.axvline(x=i - match_part_home.first_valid_index(), color="darkred", ls="--")

    ax2 = ax.twinx()
    print(len(x_axis), len(y_running_home))
    sns.lineplot(ax=ax2, x=x_axis, y=y_running_home, color="blue", label="Home Predictions")
    sns.lineplot(ax=ax2, x=x_axis, y=-y_running_away, color="red", label="Away Predictions")
    sns.lineplot(ax=ax2, x=x_axis, y=y_running_home - y_running_away, color="gray",
                 label="Prediction Difference")

    return fig


def compute_running_mean(values, window_length_lookback, window_length_outlook, hidden):
    values = np.pad(values.flatten(), (window_length_lookback - 1, window_length_outlook - 1), 'constant',
                    constant_values=(0, 0))
    running_mean_array = np.array(
        [values[i:i + window_length_outlook].sum() / min(window_length_outlook, i + 1,
                                                         len(values) - window_length_outlook - i + 1)
         for i in range(len(values) - window_length_outlook + 1)])

    return np.concatenate([np.zeros(window_length_lookback + hidden), running_mean_array])
