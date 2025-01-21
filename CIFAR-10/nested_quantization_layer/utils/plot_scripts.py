import logging
import os
from typing import List, Tuple, Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

matplotlib.use("Agg")


def process_file_logged_per_epoch(file_path: str) -> List[List[float]]:
    """
    Parses a file to extract epoch-wise logged values and organizes them by epoch.
    By epoch-wise, we mean these values have entries "Epoch {number}"
    """
    all_values = []
    epochs = []
    current_values = []

    with open(file_path) as file:
        for line in file:
            line = line.strip()

            if line.startswith("Epoch"):
                current_epoch = int(line.split()[1])
                epochs.append(current_epoch)

                if current_epoch != 0:
                    all_values.append(current_values)
                    current_values = []
            else:
                value = float(line)
                current_values.append(value)

    all_values.append(current_values)
    return all_values


def process_file_logged_without_epoch(file_path: str) -> Tuple[List[float], List[Any]]:
    """
    Parses a file to extract values and their occurrences without epoch information.
    This is applicable to values looged at the end of training, for example.
    """
    current_values = []
    val_occurences = []

    with open(file_path) as file:
        for line in file:
            line = line.split(", ")

            value = float(line[0])
            if len(line) > 1:
                occurences = int(line[1])
                val_occurences.append(occurences)

            current_values.append(value)

    return current_values, val_occurences


def plot_values_logged_on_epoch_end(log_dir_path: str) -> None:
    logging.info(
        "Generating plots for metrics logged at the end of each epoch, excluding accuracy and loss..."
    )

    on_epoch_end_dir = os.path.join(log_dir_path, "on_epoch_end")

    if not os.path.exists(on_epoch_end_dir):
        logging.warning(f"There's no on_epoch_end folder in {log_dir_path}...Skipping.")
        return

    log_files = os.listdir(on_epoch_end_dir)
    plot_dir = os.path.join(log_dir_path, "plots")

    for j, log_file in enumerate(log_files):
        if "Number" not in log_file:
            continue
        log_img_title = log_file.split(".")[0]

        if os.path.exists(plot_dir + "/" + log_img_title + ".png"):
            logging.info(
                f"{j+1}. The plot for log file {log_file} was already generated. Delete it and run the script to regenerate."
            )
            #continue

        logging.info(f"{j+1}-a. Generating plot for log file {log_file}...")

        log_file_path = os.path.join(on_epoch_end_dir, log_file)

        all_values = process_file_logged_per_epoch(log_file_path)

        plt.figure(figsize=(6, 6))
        for i in range(len(all_values[0])):
            scale_trajectory = [epoch[i] for epoch in all_values]
            plt.plot(
                range(1, len(all_values) + 1),
                scale_trajectory,
                linestyle="-",
                marker="o",
                label=f"scale_w_{i}",
                linewidth=5,
                markersize=10,
                color="orange"
            )

        plt.xlabel("", fontsize=20, labelpad=15)

        plt.ylabel("", fontsize=20)
        #plt.title(f"{log_img_title}")

        y_ticks = range(0, 131, 10)
        plt.yticks(list(y_ticks), fontsize=20)

        ticks = range(2, len(all_values) + 1, 2)
        plt.xticks(list(ticks), fontsize=20, rotation=45)

        plt.grid(True)

        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"{log_img_title}.png"), transparent=True)

        plt.close()


        logging.info(
            f"{j+1}-b. Plot for log file {log_file} was successfully generated."
        )


def plot_accuracy_per_epoch(log_dir_path: str) -> None:
    """
    This is applicable to only one training scenario.
    """

    logging.info("Generating plot for training and validation accuracy...")

    on_epoch_end_dir = os.path.join(log_dir_path, "accuracy")
    log_files = os.listdir(on_epoch_end_dir)
    plot_dir = os.path.join(log_dir_path, "plots")

    plt.figure(figsize=(12, 8))

    if os.path.exists(plot_dir + "/" + "Accuracy.png"):
        return

    for log_file in log_files:

        log_file.split(".")[0]
        log_file_path = f"{on_epoch_end_dir}/{log_file}"

        all_values = process_file_logged_per_epoch(log_file_path)

        for i in range(len(all_values[0])):
            scale_trajectory = [epoch[i] for epoch in all_values]
            plt.plot(
                range(1, len(all_values) + 1),
                scale_trajectory,
                linestyle="-",
                marker="o",
                label=f"{log_file}",
            )

    plt.xlabel("Epochs")
    plt.ylabel("Values")
    plt.title("Accuracy")
    plt.xticks(range(1, len(all_values) + 1), rotation=90)

    plt.legend(loc="best", fontsize="small", title="Legend")
    plt.grid(True)

    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "Accuracy.png"))

    plt.close()

    logging.info(
        "Plot for training and validation accuracy was successfully generated."
    )


def plot_total_loss_per_epoch(log_dir_path: str) -> None:
    """
    This is applicable to only one training scenario.
    """

    logging.info("Generating plot for training and validation loss...")

    on_epoch_end_dir = os.path.join(log_dir_path, "loss")
    log_files = os.listdir(on_epoch_end_dir)
    plot_dir = os.path.join(log_dir_path, "plots")

    plt.figure(figsize=(12, 8))

    if os.path.exists(plot_dir + "/" + "Total_loss.png"):
        return

    for log_file in log_files:

        log_file_path = os.path.join(on_epoch_end_dir, log_file)

        all_values = process_file_logged_per_epoch(log_file_path)

        for i in range(len(all_values[0])):
            scale_trajectory = [epoch[i] for epoch in all_values]
            plt.plot(
                range(1, len(all_values) + 1),
                scale_trajectory,
                linestyle="-",
                marker="o",
                label=f"{log_file}",
            )

    plt.xlabel("Epochs")
    plt.ylabel("Values")
    plt.title("Total Loss")
    plt.xticks(range(1, len(all_values) + 1), rotation=90)

    plt.legend(loc="best", fontsize="small", title="Legend")

    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "Total_loss.png"))

    plt.close()

    logging.info("Plot for training and validation loss was successfully generated.")


def unique_values_on_end(log_dir_path: str) -> int:
    """
    Counts the unique values logged at the end of training from all relevant log files.
    """
    on_train_end_dir = os.path.join(log_dir_path, "on_train_end")
    log_files = os.listdir(on_train_end_dir)

    unique_values = []

    for log_file in log_files:
        if "Unique" in log_file:
            log_file_path = os.path.join(on_train_end_dir, log_file)
            values, _ = process_file_logged_without_epoch(log_file_path)
            unique_values += values

    unique_vals = list(set(unique_values))

    return len(unique_vals)


def value_range_on_end(log_dir_path: str) -> int:
    """
    Computes the maximum range (absolute value) of parameters logged at the end of training.
    """
    on_train_end_dir = os.path.join(log_dir_path, "on_train_end")
    log_files = os.listdir(on_train_end_dir)

    max_range_over_params = 0

    for log_file in log_files:
        if "Unique" in log_file:
            log_file_path = os.path.join(on_train_end_dir, log_file)
            values, _ = process_file_logged_without_epoch(log_file_path)
            value_range = np.max(np.abs(values))

            if value_range > max_range_over_params:
                max_range_over_params = value_range

    return max_range_over_params


def final_val_acc_on_end(log_dir_path: str) -> int:
    """
    Retrieves the final validation accuracy from the log file.
    """
    val_accuracy_log = os.path.join(log_dir_path, "accuracy", "val_accuracy.log")
    accuracy_over_epochs = process_file_logged_per_epoch(val_accuracy_log)
    final_val_accuracy = accuracy_over_epochs[-1][0]

    return final_val_accuracy


def final_val_loss_on_end(log_dir_path: str) -> int:
    """
    Retrieves the final validation loss from the log file.
    """
    val_loss_log = os.path.join(log_dir_path, "loss", "val_loss.log")
    losses_over_epochs = process_file_logged_per_epoch(val_loss_log)
    final_val_loss = losses_over_epochs[-1][0]

    return final_val_loss


def pareto_unique_values_accuracy(scenario_dir_path: str) -> None:

    logging.info(
        "Starting Pareto plot generation: analyzing trade-off between unique values and model validation accuracy..."
    )

    log_dirs = [
        dir
        for dir in os.listdir(scenario_dir_path)
        if os.path.isdir(os.path.join(scenario_dir_path, dir)) and dir != "artefacts"
    ]
    plot_dir = os.path.join(scenario_dir_path, "pareto_plots")

    unique_values_list = []
    final_val_accuracy_list = []
    penalty_rate_list = []

    for log_dir in log_dirs:
        if "plots" not in log_dir:
            penalty_rate = log_dir.split("_")[-1]
            log_dir_path = f"{scenario_dir_path}/{log_dir}"

            try:
                unique_values = unique_values_on_end(log_dir_path)
                final_val_accuracy = final_val_acc_on_end(log_dir_path)
                unique_values_list.append(unique_values)
                final_val_accuracy_list.append(final_val_accuracy)
                penalty_rate_list.append(float(penalty_rate))

            except:
                logging.warning(
                    f"Something went wrong in trying to process {log_dir_path}. Maybe it's a baseline scenario and doesn't have the on_epoch_end folder...Skipping."
                )
                continue

    sorted_indices = sorted(
        range(len(penalty_rate_list)), key=lambda i: penalty_rate_list[i]
    )
    sorted_penalty_rate_list = [penalty_rate_list[i] for i in sorted_indices]
    sorted_unique_values_list = [unique_values_list[i] for i in sorted_indices]
    sorted_final_val_accuracy_list = [
        final_val_accuracy_list[i] for i in sorted_indices
    ]

    plt.figure(figsize=(10, 10))
    plt.plot(
        sorted_final_val_accuracy_list,
        sorted_unique_values_list,
        linestyle="-",
        marker="o",
        label="Pareto Curve",
    )
    for i, label in enumerate(sorted_penalty_rate_list):
        plt.annotate(
            label,
            (sorted_final_val_accuracy_list[i], sorted_unique_values_list[i]),
            fontsize=12,
        )

    plt.xlabel("Final Validation Accuracy")
    plt.ylabel("Unique Values")
    plt.title("Pareto Plot of Unique Values vs. Accuracy")
    plt.yscale("log")
    plt.grid(True)

    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"Pareto_unique_values_accuracy.png"))

    plt.close()

    logging.info(
        "Pareto plot with trade-off between unique values and model validation accuracy completed successfully."
    )


def pareto_unique_values_loss(scenario_dir_path: str) -> None:

    logging.info(
        "Starting Pareto plot generation: analyzing trade-off between unique values and model validation loss..."
    )

    log_dirs = [
        dir
        for dir in os.listdir(scenario_dir_path)
        if os.path.isdir(os.path.join(scenario_dir_path, dir)) and dir != "artefacts"
    ]
    plot_dir = os.path.join(scenario_dir_path, "pareto_plots")

    unique_values_list = []
    final_val_loss_list = []
    penalty_rate_list = []

    for log_dir in log_dirs:
        if "plots" not in log_dir:
            penalty_rate = log_dir.split("_")[-1]
            log_dir_path = os.path.join(scenario_dir_path, log_dir)

            try:
                unique_values = unique_values_on_end(log_dir_path)
                final_val_loss = final_val_loss_on_end(log_dir_path)

                unique_values_list.append(unique_values)
                final_val_loss_list.append(final_val_loss)
                penalty_rate_list.append(
                    float(penalty_rate)
                )  # Convert penalty rate to float for sorting

            except:
                logging.warning(
                    f"Something went wrong in trying to process {log_dir_path}. Maybe it's a baseline scenario...Skipping."
                )
                continue

    sorted_indices = sorted(
        range(len(penalty_rate_list)), key=lambda i: penalty_rate_list[i]
    )
    sorted_penalty_rate_list = [penalty_rate_list[i] for i in sorted_indices]
    sorted_unique_values_list = [unique_values_list[i] for i in sorted_indices]
    sorted_final_val_loss_list = [final_val_loss_list[i] for i in sorted_indices]

    plt.figure(figsize=(10, 10))
    plt.plot(
        sorted_final_val_loss_list,
        sorted_unique_values_list,
        linestyle="-",
        marker="o",
        label="Pareto Curve",
    )
    for i, label in enumerate(sorted_penalty_rate_list):
        plt.annotate(
            label,
            (sorted_final_val_loss_list[i], sorted_unique_values_list[i]),
            fontsize=12,
        )  # Add labels

    plt.xlabel("Final Validation Loss")
    plt.ylabel("Unique Values")
    plt.title("Pareto Plot of Unique Values vs. Loss")
    plt.yscale("log")
    plt.grid(True)

    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"Pareto_unique_values_loss.png"))

    plt.close()

    logging.info(
        "Pareto plot with trade-off between unique values and model validation loss completed successfully."
    )


def pareto_range_accuracy(scenario_dir_path: str) -> None:

    logging.info(
        "Starting Pareto plot generation: analyzing trade-off between range of values and model validation accuracy..."
    )

    log_dirs = [
        dir
        for dir in os.listdir(scenario_dir_path)
        if os.path.isdir(os.path.join(scenario_dir_path, dir)) and dir != "artefacts"
    ]
    plot_dir = os.path.join(scenario_dir_path, "pareto_plots")

    range_list = []
    final_val_accuracy_list = []
    penalty_rate_list = []

    for log_dir in log_dirs:
        if "plots" not in log_dir:
            penalty_rate = log_dir.split("_")[-1]
            log_dir_path = os.path.join(scenario_dir_path, log_dir)

            try:
                value_range = value_range_on_end(log_dir_path)
                final_val_accuracy = final_val_acc_on_end(log_dir_path)

                range_list.append(value_range)
                final_val_accuracy_list.append(final_val_accuracy)
                penalty_rate_list.append(float(penalty_rate))

            except:
                logging.warning(
                    f"Something went wrong in trying to process {log_dir_path}. Maybe it's a baseline scenario...Skipping."
                )
                continue

    sorted_indices = sorted(
        range(len(penalty_rate_list)), key=lambda i: penalty_rate_list[i]
    )
    sorted_penalty_rate_list = [penalty_rate_list[i] for i in sorted_indices]
    sorted_range_list = [range_list[i] for i in sorted_indices]
    sorted_final_val_accuracy_list = [
        final_val_accuracy_list[i] for i in sorted_indices
    ]

    plt.figure(figsize=(10, 10))
    plt.plot(
        sorted_final_val_accuracy_list,
        sorted_range_list,
        linestyle="-",
        marker="o",
        label="Pareto Curve",
    )
    for i, label in enumerate(sorted_penalty_rate_list):
        plt.annotate(
            label,
            (sorted_final_val_accuracy_list[i], sorted_range_list[i]),
            fontsize=12,
        )

    plt.xlabel("Final Validation Accuracy")
    plt.ylabel("Range (maxbin)")
    plt.title("Pareto Plot of Range (maxbin) vs. Accuracy")
    plt.yscale("log")
    plt.grid(True)

    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"Pareto_range_accuracy.png"))

    plt.close()

    logging.info(
        "Pareto plot with trade-off between range of values and model validation accuracy completed successfully."
    )


def pareto_range_loss(scenario_dir_path: str) -> None:

    logging.info(
        "Starting Pareto plot generation: analyzing trade-off between range of values and model validation loss..."
    )

    log_dirs = [
        dir
        for dir in os.listdir(scenario_dir_path)
        if os.path.isdir(os.path.join(scenario_dir_path, dir)) and dir != "artefacts"
    ]
    plot_dir = os.path.join(scenario_dir_path, "pareto_plots")

    range_list = []
    final_val_loss_list = []
    penalty_rate_list = []

    for log_dir in log_dirs:
        if "plots" not in log_dir:
            penalty_rate = log_dir.split("_")[-1]
            log_dir_path = os.path.join(scenario_dir_path, log_dir)

            try:
                value_range = value_range_on_end(log_dir_path)
                final_val_loss = final_val_loss_on_end(log_dir_path)

                range_list.append(value_range)
                final_val_loss_list.append(final_val_loss)
                penalty_rate_list.append(float(penalty_rate))
            except:
                logging.warning(
                    f"Something went wrong in trying to process {log_dir_path}. Maybe it's a baseline scenario...Skipping."
                )
                continue

    sorted_indices = sorted(
        range(len(penalty_rate_list)), key=lambda i: penalty_rate_list[i]
    )
    sorted_penalty_rate_list = [penalty_rate_list[i] for i in sorted_indices]
    sorted_range_list = [range_list[i] for i in sorted_indices]
    sorted_final_val_loss_list = [final_val_loss_list[i] for i in sorted_indices]

    plt.figure(figsize=(10, 10))
    plt.plot(
        sorted_final_val_loss_list,
        sorted_range_list,
        linestyle="-",
        marker="o",
        label="Pareto Curve",
    )
    for i, label in enumerate(sorted_penalty_rate_list):
        plt.annotate(
            label, (sorted_final_val_loss_list[i], sorted_range_list[i]), fontsize=12
        )  # Add labels

    plt.xlabel("Final Validation Loss")
    plt.ylabel("Range (maxbin)")
    plt.title("Pareto Plot of Range (maxbin) vs. Loss")
    plt.yscale("log")
    plt.grid(True)

    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"Pareto_range_loss.png"))

    plt.close()

    logging.info(
        "Pareto plot with trade-off between range of values and model validation loss completed successfully."
    )


def plot_histograms_on_train_end(log_dir_path: str) -> None:
    logging.info(
        "Generating plots for values logged at the end of training. These will result in histograms..."
    )    

    on_train_end_dir = os.path.join(log_dir_path, "on_train_end")

    if not os.path.exists(on_train_end_dir):
        logging.warning(f"There's no on_train_end folder in {log_dir_path}...Skipping.")
        return

    log_files = os.listdir(on_train_end_dir)
    plot_dir = os.path.join(log_dir_path, "plots")

    for j, log_file in enumerate(log_files):

        log_img_title = log_file.split(".")[0]

        if os.path.exists(plot_dir + "/" + log_img_title + ".png"):
            logging.info(
                f"{j+1}. The plot for log file {log_file} was already generated. Delete it and run the script to regenerate."
            )
            #continue

        logging.info(f"{j+1}-a. Generating plot for log file {log_file}...")

        log_file_path = os.path.join(on_train_end_dir, log_file)

        all_values, their_occurrences = process_file_logged_without_epoch(log_file_path)

        if their_occurrences == []:
            logging.info(
                f"{j+1}. The histogram for log file {log_file} doesn't make sense, as there are no number of occurences logged."
            )
            continue

        plt.figure(figsize=(10,10))
        plt.bar(all_values, their_occurrences, color="orange")

        plt.xlabel('Unique value')
        plt.ylabel('Number of occurences')
        plt.title('')

        plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)

        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"{log_img_title}.png"), transparent=False)

        plt.close()

        logging.info(
            f"Histogram plot for {log_file} generated successfully."
        )


def plot_histograms_on_train_begin(log_dir_path: str) -> None:
    logging.info(
        "Generating plots for values logged at the beginning of training. These will result in histograms..."
    )    

    on_train_begin_dir = os.path.join(log_dir_path, "on_train_begin")

    if not os.path.exists(on_train_begin_dir):
        logging.warning(f"There's no on_train_begin folder in {log_dir_path}...Skipping.")
        return

    log_files = os.listdir(on_train_begin_dir)
    plot_dir = os.path.join(log_dir_path, "plots")

    for j, log_file in enumerate(log_files):

        log_img_title = log_file.split(".")[0]

        if os.path.exists(plot_dir + "/" + log_img_title + ".png"):
            logging.info(
                f"{j+1}. The plot for log file {log_file} was already generated. Delete it and run the script to regenerate."
            )
            continue

        logging.info(f"{j+1}-a. Generating plot for log file {log_file}...")

        log_file_path = os.path.join(on_train_begin_dir, log_file)

        all_values, their_occurrences = process_file_logged_without_epoch(log_file_path)

        if their_occurrences == []:
            logging.info(
                f"{j+1}. The histogram for log file {log_file} doesn't make sense, as there are no number of occurences logged."
            )
            continue

        plt.figure(figsize=(10, 10))
        plt.bar(all_values, their_occurrences, color="orange")

        plt.xlabel('Unique value')
        plt.ylabel('Number of occurences')
        plt.title('')

        plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)

        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"{log_img_title}.png"))

        plt.close()

        logging.info(
            f"Histogram plot for {log_file} generated successfully."
        )


def val_acc_over_penalty(scenario_dir_path: str) -> None:
    """
    Generates plot for validation accuracy over epochs, including all training scenarios in the given dir.
    """

    logging.info(
        "Starting val acc over penalty plot generation..."
    )

    log_dirs = [
        dir
        for dir in os.listdir(scenario_dir_path)
        if os.path.isdir(os.path.join(scenario_dir_path, dir)) and dir != "artefacts"
    ]    

    plot_dir = os.path.join(scenario_dir_path, "pareto_plots")

    all_val_accs = []
    penalty_rates = []

    for log_dir in log_dirs:
        if "plots" not in log_dir:
            penalty_rate = log_dir.split("_")[-1]
            penalty_rates.append(penalty_rate)
            log_dir_path = os.path.join(scenario_dir_path, log_dir)

            try:
                val_acc_file = os.path.join(log_dir_path, "accuracy", "val_accuracy.log")
                val_acc_over_epochs = process_file_logged_per_epoch(val_acc_file)
                all_val_accs.append(val_acc_over_epochs)

            except:
                logging.warning(
                    f"Something went wrong in trying to process {log_dir_path}. Maybe you didn't finish training...Skipping."
                )
                continue
    
    plt.figure(figsize=(10, 10))

    sorted_indices = sorted(
        range(len(penalty_rates)), key=lambda i: penalty_rates[i]
    )
    sorted_penalty_rates = [penalty_rates[i] for i in sorted_indices]
    penalty_rates = [sorted_penalty_rates[0]] + sorted_penalty_rates[1:][::-1]
    sorted_val_accs = [all_val_accs[i] for i in sorted_indices]
    all_val_accs = [sorted_val_accs[0]] + sorted_val_accs[1:][::-1]

    for i, val_accs in enumerate(all_val_accs):
        for j in range(len(val_accs[0])):
            scale_trajectory = [epoch[j] for epoch in val_accs]
            plt.plot(
                range(1, len(val_accs) + 1),
                scale_trajectory,
                linestyle="-",
                marker="o",
                label=f"{penalty_rates[i]}",
                linewidth=5,
                markersize=10,
            )

    plt.title("Validation Accuracy over Epochs", fontsize=20, pad=40)

    plt.ylabel("Validation Accuracy", fontsize=20)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    yticks = [0] + [i / 100 for i in range(10, 101, 10)]
    plt.yticks(yticks, fontsize=20) 
    plt.tick_params(axis='y', labelsize=20)

    plt.xlabel("Epoch", fontsize=20, labelpad=15)
    plt.xticks(range(1, len(val_acc_over_epochs) + 1), rotation=30, fontsize=20)
    
    plt.legend(loc="lower left", fontsize=20, title="λ", title_fontsize=20)

    plt.grid(True)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "Val_accuracy.png"), transparent=False)

    plt.close()

    logging.info(
        "Combined plot for validation accuracy was successfully generated."
    )


def val_loss_over_penalty(scenario_dir_path: str) -> None:
    """
    Generates plot for validation loss over epochs, including all training scenarios in the given dir.
    """

    logging.info(
        "Starting val loss over penalty plot generation..."
    )

    log_dirs = [
        dir
        for dir in os.listdir(scenario_dir_path)
        if os.path.isdir(os.path.join(scenario_dir_path, dir)) and dir != "artefacts"
    ]    

    plot_dir = os.path.join(scenario_dir_path, "pareto_plots")

    all_val_losses = []
    penalty_rates = []

    for log_dir in log_dirs:
        if "plots" not in log_dir:
            penalty_rate = log_dir.split("_")[-1]
            penalty_rates.append(penalty_rate)
            log_dir_path = os.path.join(scenario_dir_path, log_dir)

            try:
                val_loss_file = os.path.join(log_dir_path, "loss", "val_loss.log")
                val_loss_over_epochs = process_file_logged_per_epoch(val_loss_file)
                all_val_losses.append(val_loss_over_epochs)

            except:
                logging.warning(
                    f"Something went wrong in trying to process {log_dir_path}. Maybe you didn't finish training...Skipping."
                )
                continue
    
    plt.figure(figsize=(10, 10))

    sorted_indices = sorted(
        range(len(penalty_rates)), key=lambda i: penalty_rates[i]
    )
    sorted_penalty_rates = [penalty_rates[i] for i in sorted_indices]
    penalty_rates = [sorted_penalty_rates[0]] + sorted_penalty_rates[1:][::-1]
    sorted_val_losses = [all_val_losses[i] for i in sorted_indices]
    all_val_losses = [sorted_val_losses[0]] + sorted_val_losses[1:][::-1]

    for i, val_accs in enumerate(all_val_losses):
        for j in range(len(val_accs[0])):
            scale_trajectory = [epoch[j] for epoch in val_accs]
            plt.plot(
                range(1, len(val_accs) + 1),
                scale_trajectory,
                linestyle="-",
                marker="o",
                label=f"{penalty_rates[i]}",
                linewidth=5,
                markersize=10
            )

    plt.title("Validation Loss over Epochs", fontsize=20, pad=40)

    plt.ylabel("Validation loss", fontsize=20)
    plt.yticks(fontsize=20) 
    plt.tick_params(axis='y', labelsize=20)

    plt.xlabel("Epoch", fontsize=20, labelpad=15)
    plt.xticks(range(1, len(val_loss_over_epochs) + 1), rotation=30, fontsize=20)
    
    plt.legend(loc="upper left", fontsize=20, title="λ", title_fontsize=20)

    plt.grid(True)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "Val_loss.png"), transparent=False)

    plt.close()

    logging.info(
        "Combined plot for validation loss was successfully generated."
    )