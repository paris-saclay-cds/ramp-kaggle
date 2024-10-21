import click
import click_config_file
import re
import json
import shutil
import numpy as np
import pandas as pd
import rampwf as rw
import rampds as rs
import rampka as rk

from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.rest import ApiException
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

def extract_kaggle_submission_info(f_name):
    pattern = r'auto_v(\d+_\d+)_n(\d+)_(bagged_then_blended|last_blend)_(\d+)(?:_r(\d+))?\.csv'
    match = re.search(pattern, f_name)
    kaggle_submission_info = {
        "version": None,
        "number": None,
        "blend_type": "not_a_blend",
        "n_folds_final_blend": None,
        "round_idx": -1  # Default value if no round_idx is found
    }

    if match:
        kaggle_submission_info['version'] = match.group(1)
        kaggle_submission_info['number'] = match.group(2)
        kaggle_submission_info['blend_type'] = "bagged_then_blended" if match.group(3) == "bagged_then_blended" else "blended_then_bagged"
        kaggle_submission_info['n_folds_final_blend'] = int(match.group(4))

        # Check if there is a round_idx
        round_idx = match.group(5)
        if round_idx:
            kaggle_submission_info['round_idx'] = int(round_idx)
    return kaggle_submission_info


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
    "--n-folds-hyperopt",
    default=3,
    show_default=True,
    help="The number of folds used in hyperopt.",
)
@click.option(
    "--n-folds-final-blend",
    default=30,
    show_default=True,
    help="The number of folds to bag",
)
@click.option(
    "--first-fold-idx",
    default=0,
    show_default=True,
    help="The index of the first fold of problem.get_cv.",
)
@click_config_file.configuration_option()
def main(
    ramp_kit,
    version,
    number,
    n_folds_hyperopt,
    n_folds_final_blend,
    first_fold_idx,
):
    round_idxs = [20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000]

    ramp_kit_dir = f"{ramp_kit}_v{version}_n{number}"
    problem = rw.utils.assert_read_problem(ramp_kit_dir=ramp_kit_dir)
    score_names = [st.name for st in problem.score_types]
    score_type = problem.score_types[-1]
    valid_score_name = f'valid_{score_names[-1]}'
    metadata = json.load(open(Path(ramp_kit_dir) / "data" / "metadata.json"))
    competition = metadata["kaggle"]["name"]

    all_actions = rs.actions.get_all_actions(ramp_kit_dir)
    blend_actions = [ra for ra in all_actions if ra.name == "blend" and
                     ra.kwargs["fold_idxs"] == range(first_fold_idx, first_fold_idx + n_folds_hyperopt)]
    # adding possible final blend, not in the scheduled round_idx's
    if len(blend_actions) not in round_idxs:
        round_idxs = round_idxs + [len(blend_actions)]
        round_idxs = [idx for idx in round_idxs if idx <= len(blend_actions)]

    kaggle_results_df = rk.actions.get_kaggle_results(competition)
    try:
        kaggle_submission_infos = [extract_kaggle_submission_info(f_name) for f_name in kaggle_results_df["file_name"]]
        kaggle_results_df = pd.concat([kaggle_results_df, pd.DataFrame(kaggle_submission_infos)], axis=1)
        kaggle_results_df = kaggle_results_df[kaggle_results_df["round_idx"].isin(round_idxs)]
        kaggle_results_df = kaggle_results_df[kaggle_results_df["version"] == version]
        kaggle_results_df = kaggle_results_df[kaggle_results_df["number"] == number]
        kaggle_results_df = kaggle_results_df[kaggle_results_df["round_idx"] > 0]
        kaggle_results_df["ramp_kit"] = ramp_kit
        results_path = Path(ramp_kit_dir) / "results"
        results_path.mkdir(exist_ok=True)

        save_df = kaggle_results_df[kaggle_results_df["blend_type"] == "blended_then_bagged"]
        save_df = save_df.drop_duplicates(subset='round_idx', keep='last')
        save_df = save_df.sort_values("round_idx")
        save_df.to_csv(results_path / "kaggle_results_blended_then_bagged.csv", index=False)
        print(f"saved blended_then_bagged round idxs: {list(save_df['round_idx'])}")

        save_df = kaggle_results_df[kaggle_results_df["blend_type"] == "bagged_then_blended"]
        save_df = save_df.drop_duplicates(subset='round_idx', keep='last')
        save_df = save_df.sort_values("round_idx")
        save_df.to_csv(results_path / "kaggle_results_bagged_then_blended.csv", index=False)
        print(f"saved bagged_then_blended round idxs: {list(save_df['round_idx'])}")
    except Exception as e:
        print(e)

def start():
    main()

if __name__ == "__main__":
    start()
