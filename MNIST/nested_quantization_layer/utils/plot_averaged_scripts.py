import argparse
import os
from typing import Dict, List
import numpy as np

import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

import plot_scripts


def _plot_pareto_curve(
    x_data: List[float],
    y_data: List[float],
    x_label: str,
    y_label: str,
    title: str,
    filename: str,
    penalty_rate_list: str,
    scale_app_scenario_dir: str,
) -> None:

    plt.figure(figsize=(6, 6))
    plt.plot(x_data, y_data, linestyle="-", marker="o", label="Pareto Curve", linewidth=5, markersize=10, color="orange")

    for i, label in enumerate(penalty_rate_list):
        continue
        if i == 0:
            plt.annotate(f"λ={label}", (x_data[i], y_data[i]),xytext=(-65, -10),textcoords='offset points',  fontsize=20)  # Annotate each point
        elif i == 1:
            plt.annotate(f"λ={label}", (x_data[i], y_data[i]),xytext=(-90, -10),textcoords='offset points',  fontsize=20)  # Annotate each point
        elif i == 2:
            plt.annotate(f"λ={label}", (x_data[i], y_data[i]),xytext=(-90, -5),textcoords='offset points',  fontsize=20)  # Annotate each point
        elif i == 3:
            plt.annotate(f"λ={label}", (x_data[i], y_data[i]),xytext=(-65, -30),textcoords='offset points',  fontsize=20)  # Annotate each point
        elif i == 4:
            plt.annotate(f"λ={label}", (x_data[i], y_data[i]),xytext=(5, 10),textcoords='offset points',  fontsize=20)  # Annotate each point
        elif i == 5:
            plt.annotate(f"λ={label}", (x_data[i], y_data[i]),xytext=(-12, 10),textcoords='offset points',  fontsize=20)  # Annotate each point
        elif i == 6:
            plt.annotate(f"λ={label}", (x_data[i], y_data[i]),xytext=(-12, -14),textcoords='offset points',  fontsize=20)  # Annotate each point

    for i, label in enumerate(penalty_rate_list):
        plt.annotate(f"", (x_data[i], y_data[i]),xytext=(-65, -10),textcoords='offset points',  fontsize=20)  # Annotate each point

    plt.xlabel("Unique Integers", fontsize=20)
    plt.ylabel("", fontsize=20)

    x_ticks = np.arange(0.15, 1.05, 0.1)  # Generate ticks from 0 to 1 at 0.05 intervals
    plt.xticks(x_ticks, rotation=45)
    plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    plt.xlim((0.15, 1.0))

    y_ticks = [1, 10, 100, 1_000, 10_000, 100_000]
    y_lim = (1, 30_000)
    plt.yticks(y_ticks)

    plt.tick_params(axis='x', labelsize=19)
    plt.tick_params(axis='y', labelsize=20)
    
#    plt.title(title, fontsize=15)
    plt.yscale("log")
    plt.ylim(y_lim)
#    plt.grid(True)

    file_path = os.path.join(scale_app_scenario_dir, filename)
    plt.savefig(file_path, transparent=True)
    plt.close()


def averaged_pareto_unique_values_accuracy(scale_app_scenario_dir: str) -> None:
    def calculate_average(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0

    def process_train_scenario(train_scenario: str, results: Dict) -> None:
        penalty_rate = float(train_scenario.split("_")[-1])
        unique_values = plot_scripts.unique_values_on_end(train_scenario)
        value_range = plot_scripts.value_range_on_end(train_scenario)
        final_val_accuracy = plot_scripts.final_val_acc_on_end(train_scenario)
        final_val_loss = plot_scripts.final_val_loss_on_end(train_scenario)
        results.setdefault(
            penalty_rate,
            {
                "unique_values_over_seeds": [],
                "value_ranges_over_seeds": [],
                "final_val_acc_over_seeds": [],
                "final_val_loss_over_seeds": [],
            },
        )
        results[penalty_rate]["unique_values_over_seeds"].append(unique_values)
        results[penalty_rate]["value_ranges_over_seeds"].append(value_range)
        results[penalty_rate]["final_val_acc_over_seeds"].append(final_val_accuracy)
        results[penalty_rate]["final_val_loss_over_seeds"].append(final_val_loss)

    results = {}

    seeds = [
        os.path.join(scale_app_scenario_dir, dir)
        for dir in os.listdir(scale_app_scenario_dir)
        if os.path.isdir(os.path.join(scale_app_scenario_dir, dir))
    ]

    for seed in seeds:
        train_scenarios = [
            os.path.join(seed, dir)
            for dir in os.listdir(seed)
            if dir not in ["artefacts", "pareto_plots", "plotting_process.log"]
            and os.path.isdir(os.path.join(seed, dir))
        ]

        for train_scenario in train_scenarios:
            process_train_scenario(train_scenario, results)

    penalty_rate_list = sorted(results.keys())
    unique_values_list = [
        calculate_average(results[penalty_rate]["unique_values_over_seeds"])
        for penalty_rate in penalty_rate_list
    ]
    value_ranges_list = [
        calculate_average(results[penalty_rate]["value_ranges_over_seeds"])
        for penalty_rate in penalty_rate_list
    ]
    final_val_accuracy_list = [
        calculate_average(results[penalty_rate]["final_val_acc_over_seeds"])
        for penalty_rate in penalty_rate_list
    ]
    final_val_loss_list = [
        calculate_average(results[penalty_rate]["final_val_loss_over_seeds"])
        for penalty_rate in penalty_rate_list
    ]

    plots_info = [
        {
            "x_data": final_val_accuracy_list,
            "y_data": unique_values_list,
            "x_label": "Final Validation Accuracy",
            "y_label": "Unique Quantized Values",
            "title": "Pareto Front - Unique Quantized Values vs. Accuracy",
            "filename": "Pareto_unique_vals_accuracy.png",
        },
        {
            "x_data": final_val_loss_list,
            "y_data": unique_values_list,
            "x_label": "Final Validation Loss",
            "y_label": "Unique Quantized Values",
            "title": "Pareto Front - Unique Quantized Values vs. Loss",
            "filename": "Pareto_unique_vals_loss.png",
        },
        {
            "x_data": final_val_accuracy_list,
            "y_data": value_ranges_list,
            "x_label": "Final Validation Accuracy",
            "y_label": "Range  (max abs quantized value)",
            "title": "Pareto Front - Range (max abs quantized value) vs. Accuracy",
            "filename": "Pareto_vals_range_accuracy.png",
        },
        {
            "x_data": final_val_loss_list,
            "y_data": value_ranges_list,
            "x_label": "Final Validation Loss",
            "y_label": "Range  (max abs quantized value)",
            "title": "Pareto Plot of Range  (max abs quantized value) vs. Loss",
            "filename": "Pareto_vals_range_loss.png",
        },
    ]

    for plot_info in plots_info:
        _plot_pareto_curve(
            x_data=plot_info["x_data"],
            y_data=plot_info["y_data"],
            x_label=plot_info["x_label"],
            y_label=plot_info["y_label"],
            title=plot_info["title"],
            filename=plot_info["filename"],
            penalty_rate_list=penalty_rate_list,
            scale_app_scenario_dir=scale_app_scenario_dir,
        )


def val_acc_over_penalty(scenario_dir_path: str) -> None:

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
            penalty_rates.append[penalty_rate]
            log_dir_path = os.path.join(scenario_dir_path, log_dir)

            try:
                val_acc_file = os.path.join(log_dir_path, "accuracy", "val_accuracy.log")
                val_acc_over_epochs = plot_scripts.process_file_logged_per_epoch(val_acc_file)
                all_val_accs.append(val_acc_over_epochs)

            except:
                logging.warning(
                    f"Something went wrong in trying to process {log_dir_path}. Maybe you didn't finish training...Skipping."
                )
                continue
    
    plt.figure(figsize=(12, 8))

    for i, val_accs in enumerate(all_val_accs):
        for j in range(len(val_accs[0])):
            scale_trajectory = [epoch[j] for epoch in val_accs]
            plt.plot(
                range(1, len(val_accs) + 1),
                scale_trajectory,
                linestyle="-",
                marker="o",
                label=f"{penalty_rates[i]}",
            )

    plt.xlabel("Epochs")
    plt.ylabel("Values")
    plt.title("Validation Accuracy")
    plt.xticks(range(1, len(val_acc_over_epochs) + 1), rotation=90)

    plt.legend(loc="best", fontsize="small", title="Legend")

    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "Val_accuracy.png"))

    plt.close()

    logging.info(
        "Combined plot for validation accuracy was successfully generated."
    )

def main():
    # Parse scenario_dir argument
    parser = argparse.ArgumentParser(description="Combine results over seeds.")
    parser.add_argument(
        "scale_app_scenario_dir",
        type=str,
        help="The scale application scenario directory containing training results for different seeds.",
    )

    args = parser.parse_args()

    scale_app_scenario_dir = args.scale_app_scenario_dir

    full_path = "logs/" + scale_app_scenario_dir

    averaged_pareto_unique_values_accuracy(full_path)


if __name__ == "__main__":
    main()
