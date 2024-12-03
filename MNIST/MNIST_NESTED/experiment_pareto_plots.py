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
        ]
    )

    logging.info(f"Pareto plots...")

    plot_scripts.pareto_unique_values_accuracy(scenario_dir)
    plot_scripts.pareto_unique_values_loss(scenario_dir)
    plot_scripts.pareto_range_accuracy(scenario_dir)
    plot_scripts.pareto_range_loss(scenario_dir)

    logging.info(f"Pareto plots done.")


if __name__ == "__main__":
    main()