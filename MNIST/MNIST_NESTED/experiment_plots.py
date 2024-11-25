import os
import argparse
import logging

import utils.plot_scripts as plot_scripts

def main():
    parser = argparse.ArgumentParser(description="Create plots for specified scenario folder.")
    parser.add_argument(
        "scenario_dir", 
        type=str, 
        help="The directory containing scenario logs (e.g., 'with_maxbin_threshold_3')."
    )
    args = parser.parse_args()

    scenario_dir = args.scenario_dir
    scenario_dir = "logs/" + scenario_dir
    plotting_log_file_path = scenario_dir + "/plotting_process.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(plotting_log_file_path),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Pareto plots...")

    plot_scripts.pareto_unique_values_accuracy(scenario_dir)
    plot_scripts.pareto_unique_values_loss(scenario_dir)
    plot_scripts.pareto_range_accuracy(scenario_dir)
    plot_scripts.pareto_range_loss(scenario_dir)

    logging.info(f"Pareto plots done.")

    log_dirs = os.listdir(scenario_dir)
    log_dirs = [dir for dir in os.listdir(scenario_dir) if os.path.isdir(os.path.join(scenario_dir, dir)) and dir not in ["pareto_plots", "artefacts"]]

    for log_dir in log_dirs:
        log_dir_path = f"{scenario_dir}/{log_dir}"
        
        logging.info(f"Processing {log_dir} at path {log_dir_path}...")
        
        plot_scripts.plot_values_logged_on_epoch_end(log_dir_path)
        plot_scripts.plot_accuracy_per_epoch(log_dir_path)
        plot_scripts.plot_total_loss_per_epoch(log_dir_path)

    logging.info("Plot generation completed successfully.")

if __name__ == "__main__":
    main()