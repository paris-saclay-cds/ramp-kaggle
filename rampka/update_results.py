import click
import glob
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


# flake8: noqa: E501

def kaggle_select(kaggle_api, suffix, id):
    f_name = kaggle_api.string(getattr(id, "fileName"))
    select = f_name[5:5 + len(suffix)] == suffix
    select = select and kaggle_api.string(getattr(id, "publicScore")) != ''
    return select

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--ramp-kit",
    default=None,
    help="The kit to update.",
)
@click.option(
    "--version",
    default=None,
    help="The program version",
)
@click.option(
    "--number",
    default=None,
    help="The program number (repeated within version)",
)
@click.option(
    "--ramp-setup-dir",
    default="/nas/ramp-setup-kits",
    help="The RAMP setup dir where the leaderboards are.",
)
def main(
    ramp_kit,
    version,
    number,
    ramp_setup_dir,
):
    base_predictors = ["lgbm", "xgboost", "catboost", "skmlp", "sket", "fastai",]
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    results_summary_df = pd.read_csv("results_summary.csv")
    kaggle_ok = True
    for col in results_summary_df.columns:
        if col[:14] == "contributivity" or col[:6] == "rounds":
            results_summary_df[col] = results_summary_df[col].fillna(0)
            results_summary_df[col] = results_summary_df[col].astype("int64")
        if col[:7] == "runtime":
            results_summary_df[col] = results_summary_df[col].astype("timedelta64[ns]")
    #for row_i, row in results_summary_df.loc[[4]].iterrows():
    for row_i, row in results_summary_df.iterrows():
        if ramp_kit is not None and row["ramp_kit"] != ramp_kit:
            continue
        if version is not None and row["version"] != version:
            continue
        if number is not None and str(row["number"]) != number:
            continue
        kit_suffix = f"v{row['version']}_n{row['number']}"
        ramp_kit_dir = f"{row['ramp_kit']}_{kit_suffix}"
        print(ramp_kit_dir)
#        row.loc["run_finished"] = 0

        if row.loc["run_finished"] == 1 and row.loc["kaggle_finished"] == 1:
            continue

        problem = rw.utils.assert_read_problem(ramp_kit_dir=ramp_kit_dir)
        metadata = json.load(open(Path(ramp_kit_dir) / "data" / "metadata.json"))
        leaderboard_scores = {}
        for phase in ["public", "private"]:
            try:
                leaderboard_scores[phase] = np.load(Path(ramp_setup_dir) / row["ramp_kit"] / f"{phase}_leaderboard_scores.npy")
            except:
                pass
        available_phases = leaderboard_scores.keys()
        print(f"Available leaderboards are {available_phases}")
        n_kaggle_files = (3 + len(base_predictors)) * len(available_phases)  # growing folds, last blend, bag then blend, and best base predictors
        kaggle_file_counter = 0

        final_test_predictions_path = Path(ramp_kit_dir) / "final_test_predictions"
        action_f_names = glob.glob(f'{ramp_kit_dir}/actions/*')
        action_f_names.sort()
        ramp_program = []
        for action_f_name in action_f_names:
            f_name = Path(action_f_name).name
            ramp_program.append(rs.actions.load_ramp_action(action_f_name))

        hyperopt_actions = [ra for ra in ramp_program if ra.name == "hyperopt"]
        blend_actions = [ra for ra in ramp_program if ra.name == "blend"] + [ra for ra in ramp_program if ra.name == "bag_then_blend"]
        train_actions = [ra for ra in ramp_program if ra.name == "train"]
        kaggle_actions = [ra for ra in ramp_program if ra.name == "kaggle_submit_file" or ra.name == "submit_final_test_predictions"]
        select_top_hyperopt_actions = [ra for ra in ramp_program if ra.name == "select_top_hyperopt"]
        if len(hyperopt_actions) == 0:
            print("no hyperopt actions")
            continue

        results_summary_df.loc[row_i, "runtime_hyperopt"] = pd.to_timedelta(
            np.array([ra.runtime for ra in hyperopt_actions]).sum())
        growing_folds_start_time = max([ra.stop_time for ra in hyperopt_actions])

        results_summary_df.loc[row_i, "run_finished"] = 1

        try:
            kaggle_submission_ids = kaggle_api.competition_submissions(competition=metadata["kaggle"]["name"])
            # we select those ids that match xxxxxv<version>_n_<number>
            kaggle_submission_ids = [id for id in kaggle_submission_ids if kaggle_select(kaggle_api, kit_suffix, id)]
            kaggle_action_times = [kaggle_api.string(getattr(id, "description")) for id in kaggle_submission_ids]
            kaggle_valid_ids = np.zeros(len(kaggle_submission_ids))
            # we select those ids where the message is a valid date
            for ki in range(len(kaggle_valid_ids)):
                try:
                    pd.to_datetime(kaggle_action_times[ki])
                    kaggle_valid_ids[ki] = 1
                except:
                    pass
            kaggle_submission_ids = [id for ki, id in enumerate(kaggle_submission_ids) if kaggle_valid_ids[ki]]
            kaggle_action_times = [kaggle_api.string(getattr(id, "description")) for id in kaggle_submission_ids]
            kaggle_scores = {}
            for phase in available_phases:
                kaggle_scores[phase] = [kaggle_api.string(getattr(id, f"{phase}Score")) for id in kaggle_submission_ids]
            kaggle_file_names = [kaggle_api.string(getattr(id, "fileName")) for id in kaggle_submission_ids]
        except Exception as e:
            pass
        available_phases = kaggle_scores.keys()
        print(f"Available kaggle scores are {available_phases}")
        failure_count = 0
        no_growing_folds = False
        for blend_type in ["growing_folds", "last_blend", "bagged_then_blended"]:
            submission_file_name_1 = f"auto_{kit_suffix}_{blend_type}_931.csv"
            submission_file_name_2 = f"auto_{kit_suffix}_{blend_type}_030.csv"
            last_kaggle_action = None
            for ra in kaggle_actions:
                if ra.kwargs["submission_target_f_name"].endswith(str(final_test_predictions_path / submission_file_name_1)):
                    last_kaggle_action = ra
                    submission_file_name = submission_file_name_1
                if ra.kwargs["submission_target_f_name"].endswith(str(final_test_predictions_path / submission_file_name_2)):
                    last_kaggle_action = ra
                    submission_file_name = submission_file_name_2
            if last_kaggle_action is None:
                if blend_type == "growing_folds":
                    no_growing_folds = True
                else:
                    results_summary_df.loc[row_i, "run_finished"] = 0
                failure_count += 1
                continue
            if blend_type == "growing_folds":
                growing_folds_stop_time = last_kaggle_action.start_time
            else:
                last_blend_stop_time = last_kaggle_action.start_time
            blend_action = [ra for ra in blend_actions if ra.start_time <= last_kaggle_action.start_time][-1]
            results_summary_df.loc[row_i, f"valid_{blend_type}"] = blend_action.blended_score
            for submission in base_predictors:
                try:
                    results_summary_df.loc[row_i, f"contributivity_{blend_type}_{submission}"] =\
                        np.array([c for s, c in blend_action.contributivities.items() if s[:len(submission)] == submission]).sum()
                except AttributeError as e:
                    print("No contributivities, old bag_then_blend action")
            if submission_file_name not in kaggle_file_names:
                # We submit to Kaggle in this call, fill the table in the next, to avoid possible delays
                print(f"Submitting {submission_file_name}")
                if kaggle_ok:
                    try:
                        kaggle_api.competition_submit(
                            file_name=final_test_predictions_path / submission_file_name,
                            message=last_kaggle_action.start_time,
                            competition=metadata["kaggle"]["name"]
                        )
                    except ApiException as e:
                        print(e)
                        kaggle_ok = False
            else:
                sub_idx = kaggle_file_names.index(submission_file_name)
                for phase in available_phases:
                    score = float(kaggle_scores[phase][sub_idx])
                    results_summary_df.loc[row_i, f"kaggle_{phase}_{blend_type}"] = score
                    results_summary_df.loc[row_i, f"kaggle_{phase}_prank_{blend_type}"] = rk.actions.kaggle_prank(
                        score, leaderboard_scores[phase], problem)
                    kaggle_file_counter += 1
        if failure_count == 3:
            continue
        if not no_growing_folds:
            results_summary_df.loc[row_i, "runtime_growing_folds"] = pd.to_timedelta(
                np.array([ra.runtime for ra in train_actions if ra.start_time > growing_folds_start_time
                          and ra.start_time < growing_folds_stop_time]).sum())
        if no_growing_folds:
            n_kaggle_files -= len(available_phases)
        print(n_kaggle_files)
        # growing folds done but not last blend and bagged and blended
        if failure_count == 1 and not no_growing_folds:
            continue
        # in last blend training time, we also need to take into consideration of training time of models that occur during the growing fold iteration
        if blend_type in ["growing_folds", "last_blend"]:
            results_summary_df.loc[row_i, "runtime_last_blend"] = pd.to_timedelta(
                np.array([ra.runtime for ra in train_actions if ra.start_time > growing_folds_start_time
                          and ra.start_time < last_blend_stop_time and ra.kwargs["submission"] in blend_action.kwargs["submissions"]]).sum())
        # growing folds and last blend done but not bagged and blended
        if failure_count == 2 and not no_growing_folds:
            continue
        for submission in base_predictors:
            submission_file_name = f"auto_{kit_suffix}_best_{submission}.csv"
            submission_hyperopt_actions = [ra.runtime for ra in hyperopt_actions if ra.kwargs["submission"].startswith(submission)]
            results_summary_df.loc[row_i, f"runtime_hyperopt_{submission}"] = pd.to_timedelta(np.array(submission_hyperopt_actions).sum())
            results_summary_df.loc[row_i, f"rounds_hyperopt_{submission}"] = len(submission_hyperopt_actions)

            last_kaggle_action = None
            for ra in kaggle_actions:
                if ra.kwargs["submission_target_f_name"].endswith(str(final_test_predictions_path / submission_file_name)):
                    last_kaggle_action = ra
            if last_kaggle_action is None:
                # if contributivity is zero, it is normal not having the kaggle action
                if results_summary_df.loc[row_i, f"contributivity_last_blend_{submission}"] != 0:
                    results_summary_df.loc[row_i, "run_finished"] = 0
                else:
                    n_kaggle_files -= len(available_phases)
                continue
            select_top_hyperopt_action = [ra for ra in select_top_hyperopt_actions if ra.start_time <= last_kaggle_action.start_time][-1]
            selected_train_actions = [ra for ra in train_actions if ra.kwargs["submission"] == select_top_hyperopt_action.selected_submissions[0]]
            if len(selected_train_actions) == 0:
                n_kaggle_files -= len(available_phases)
            else:
                train_action = selected_train_actions[-1]
                for scoring_type in ["mean", "bagged"]:
                    results_summary_df.loc[row_i, f"valid_{scoring_type}_{submission}"] = train_action.__dict__[f"{scoring_type}_score"]

                if submission_file_name not in kaggle_file_names:
                    # We submit to Kaggle in this call, fill the table in the next, to avoid possible delays
                    print(f"Submitting {submission_file_name}")
                    if kaggle_ok:
                        try:
                            kaggle_api.competition_submit(
                                file_name=final_test_predictions_path / submission_file_name,
                                message=last_kaggle_action.start_time,
                                competition=metadata["kaggle"]["name"]
                            )
                        except ApiException as e:
                            print(e)
                            kaggle_ok = False
                else:
                    sub_idx = kaggle_file_names.index(submission_file_name)
                    for phase in available_phases:
                        score = float(kaggle_scores[phase][sub_idx])
                        results_summary_df.loc[row_i, f"kaggle_{phase}_{submission}"] = score
                        results_summary_df.loc[row_i, f"kaggle_{phase}_prank_{submission}"] = rk.actions.kaggle_prank(
                            score, leaderboard_scores[phase], problem)
                        kaggle_file_counter += 1

        print(n_kaggle_files, kaggle_file_counter)
        if kaggle_file_counter == n_kaggle_files:
            results_summary_df.loc[row_i, "kaggle_finished"] = 1

    shutil.copy("results_summary.csv", "results_summary_bak.csv")
    results_summary_df.to_csv("results_summary.csv", index=False)

def start():
    main()

if __name__ == "__main__":
    start()
