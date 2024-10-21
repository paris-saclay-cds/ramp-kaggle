import click
import click_config_file
import glob
import json
import shutil
import numpy as np
import pandas as pd
import rampwf as rw
import datetime

from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.rest import ApiException
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


# flake8: noqa: E501

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--ramp-kit",
    help="The kit to to submit.",
)
@click.option(
    "--version",
    help="The program version",
)
@click.option(
    "--number",
    help="The program number (repeated within version)",
)
@click.option(
    "--stop-fold-idx",
    help="The fold index",
)
@click.option(
    "--round-idx",
    default="",
    show_default=True,
    help="The race round",
)
@click.option(
    "--blend-type",
    default="last_blend",
    show_default=True,
    help="The blend type (last_blend or bagged_then_blended)",
)
@click_config_file.configuration_option()
def main(
    ramp_kit,
    version,
    number,
    stop_fold_idx,
    round_idx,
    blend_type,
):
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    kit_suffix = f"v{version}_n{number}"
    ramp_kit_dir = f"{ramp_kit}_{kit_suffix}"
    metadata = json.load(open(Path(ramp_kit_dir) / "data" / "metadata.json"))
    final_test_predictions_path = Path(ramp_kit_dir) / "final_test_predictions"
    if round_idx == "":
        submission_file_name = f"auto_{kit_suffix}_{blend_type}_{stop_fold_idx}.csv"
    else:
        submission_file_name = f"auto_{kit_suffix}_{blend_type}_{stop_fold_idx}_r{round_idx}.csv"
    try:
        kaggle_api.competition_submit(
            file_name=final_test_predictions_path / submission_file_name,
            message=datetime.datetime.utcnow(),
            competition=metadata["kaggle_name"]
        )
    except ApiException as e:
        print(e)

def start():
    main()

if __name__ == "__main__":
    start()
