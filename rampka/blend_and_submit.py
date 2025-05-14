import click
import click_config_file
import glob
import json
import time
import shutil
import numpy as np
import pandas as pd
import rampwf as rw
import rampds as rs
import rampka as rk
import datetime

from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.rest import ApiException
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

def kaggle_select(kaggle_api, suffix, id):
    f_name = kaggle_api.string(getattr(id, "fileName"))
    select = f_name[5:5 + len(suffix)] == suffix
    select = select and kaggle_api.string(getattr(id, "publicScore")) != ''
    return select

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
    "--n-folds-final-blend",
    default=30,
    show_default=True,
    help="The number of folds to bag",
)
@click.option(
    "--n-folds-hyperopt",
    default=3,
    show_default=True,
    help="The number of folds used in hyperopt.",
)
@click.option(
    "--first-fold-idx",
    default=0,
    show_default=True,
    help="The index of the first fold of problem.get_cv.",
)
@click.option(
    "--race-blend",
    default="blend",
    help="blend: first blend per fold, then bag the blends, bag_then_blend: first bag per folds, then blend the bags.",
)
@click_config_file.configuration_option()
def main(
    ramp_kit,
    version,
    number,
    n_folds_final_blend,
    n_folds_hyperopt,
    first_fold_idx,
    race_blend,
):
    round_idxs = [20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000]
    submitted_idxs = []
    kit_suffix = f"v{version}_n{number}"
    ramp_kit_dir = f"{ramp_kit}_{kit_suffix}"
    metadata = json.load(open(Path(ramp_kit_dir) / "data" / "metadata.json"))
    competition = metadata["kaggle"]["name"]
    final_test_predictions_path = Path(ramp_kit_dir) / "final_test_predictions"

    all_actions = rs.actions.get_all_actions(ramp_kit_dir)
    blend_actions = [ra for ra in all_actions if ra.name == race_blend and
                     ra.kwargs["fold_idxs"] == range(first_fold_idx, first_fold_idx + n_folds_hyperopt)]
    # adding possible final blend, not in the scheduled round_idx's
    if len(blend_actions) not in round_idxs:
        round_idxs = round_idxs + [len(blend_actions)]
        round_idxs = [idx for idx in round_idxs if idx <= len(blend_actions)]
    print(round_idxs)

    kaggle_results_df = rk.actions.get_kaggle_results(competition)

    kaggle_api = KaggleApi()
    kaggle_api.authenticate()

    for round_idx in round_idxs:
        rs.blend_at_round.blend_at_round(
            ramp_kit,
            version,
            number,
            n_folds_final_blend,
            n_folds_hyperopt,
            first_fold_idx,
            race_blend,
            round_idx,
        )
        for blend_type in ["last_blend", "bagged_then_blended"]:
            submission_file_name = f"auto_{kit_suffix}_{blend_type}_{str(first_fold_idx + n_folds_final_blend).zfill(3)}_r{round_idx}.csv"
            if len(kaggle_results_df) == 0 or (
                (final_test_predictions_path / submission_file_name).exists() and (
                     submission_file_name not in list(kaggle_results_df["file_name"]) or
                     kaggle_results_df[kaggle_results_df["file_name"] == submission_file_name]["public_score"].isna().any()
                )
            ):
                try:
                    kaggle_api.competition_submit(
                        file_name=final_test_predictions_path / submission_file_name,
                        message=datetime.datetime.utcnow(),
                        competition=competition
                    )
                    submitted_idxs.append(round_idx)
                    print("Sleeping 10 seconds ...")
                    time.sleep(10)
                except Exception as e:
                    print(e)
    print(f"Submitted rounds: {submitted_idxs}")

def start():
    main()

if __name__ == "__main__":
    start()
