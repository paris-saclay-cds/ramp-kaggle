import json
import time
import numpy as np
import rampds as rs
import rampka as rk
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import click

rs.actions.EXECUTE_PLAN = True
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


def clean_leaderboard(leaderboard_df):
    """Removes submissions with rank 0.

    Minimum rank is 1 so we remove the submissions with rank 0.

    Args:
        leaderboard_df (pd.DataFrame). Dataframe of the leaderboard.
    Returns:
        scores (np.array). Cleaned numpy array of the scores.
    """
    leaderboard_df_filtered = leaderboard_df[leaderboard_df["Rank"] != 0]

    return leaderboard_df_filtered["Score"].to_numpy()


def get_kaggle_leaderboards(ramp_kits):
    """Download Kaggle leaderboards

    Args:
        ramp_kits (list): List of the kits for which we want to download the
        leaderboards
    """
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()

    for ramp_kit in ramp_kits:
        print(ramp_kit)
        with open(Path(ramp_kit) / "metadata.json", "r") as f:
            metadata = json.load(f)

        for phase in ["private", "public"]:
            # random sleep time to appear more human if that can help (not sure...)
            time.sleep(np.random.uniform(low=0.1, high=2))
            try:
                leaderboard_scores_df = rk.actions.get_leaderboard_scores(
                    download_destination=Path(ramp_kit),
                    kaggle_api=kaggle_api,
                    competition=metadata["kaggle"]["name"],
                    phase=phase,
                )
                leaderboard_scores = clean_leaderboard(leaderboard_scores_df)
                np.save(
                    Path(ramp_kit) / f"{phase}_leaderboard_scores.npy",
                    leaderboard_scores,
                )
            except Exception as e:
                print(e)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--ramp-kit",
    help="The kit to download.",
)
def main(
    ramp_kit,
):
    get_kaggle_leaderboards([ramp_kit])


def start():
    main()

if __name__ == "__main__":
    start()

