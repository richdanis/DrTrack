import pandas as pd
import argparse

DATA_PATH = "data/05_results_Luca/"


def rename_ids(gt_df, res_df, step):
    """
    Rename ids of the result dataframe to bring them into the right format for evaluation.
    In particular, the ids of the result dataframe are renamed to match the ids of the ground truth dataframe.
    The renaming is done by comparing the x and y coordinates of the droplets in the result dataframe with the x and y coordinates of the droplets in the ground truth dataframe.
    The resulting dataframe can then be fed into the evaluation pipeline.

    Parameters
    ----------
    gt_df : pd.DataFrame
        Ground truth dataframe with high resolution data. Frequency 1.25 min.
    res_df : pd.DataFrame
        Result dataframe with low resolution data.
    step : int
        Step size between the frames in the result dataframe.

    Returns
    -------
    pd.DataFrame
        Result dataframe with renamed ids.
    """

    res_df = res_df.copy()

    # reset index
    res_df = res_df.reset_index(drop=True)

    # set max_id
    max_id = len(gt_df)

    # find max_id of res_df
    max_col = 0
    for col in res_df.columns:
        if "id_" in col:
            max_col = max(max_col, int(col.split("_")[1]))

    # iterate over all id columns in res_df
    for i in range(max_col + 1):

        x_gt, y_gt = gt_df[f"x{i*step}"].to_numpy(), gt_df[f"y{i*step}"].to_numpy()
        x_res, y_res = res_df[f"x{i}"].to_numpy(), res_df[f"y{i}"].to_numpy()

        # get pairwise distance matrix between each two points
        dist = (x_res[:, None] - x_gt) ** 2 + (y_res[:, None] - y_gt) ** 2
        mask = dist < 1e-6
        # build index mapping array, for each index in x_res map it to col index, when it has zero
        # distance with that col, if there are only false in row then assign new_id
        ind_list = []
        for j in range(len(mask)):
            if not mask[j].any():
                ind_list.append(max_id)
                max_id += 1
            else:
                ind_list.append(mask[j].argmax())

        res_df[f"id_{i}"] = ind_list

    return res_df


if __name__ == "__main__":

    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, help="Path to ground truth csv.")
    parser.add_argument("--res", type=str, help="Path to result csv.")
    parser.add_argument(
        "--step",
        type=int,
        choices=[4, 5, 12, 24],
        help="Step size between frames in result dataframe.",
    )

    args = parser.parse_args()

    # read data
    gt_df = pd.read_csv(DATA_PATH + args.gt + "/filtered_trajectories_y0_x0_basic.csv")
    res_df = pd.read_csv(
        DATA_PATH + args.res + "/filtered_trajectories_y0_x0_basic.csv"
    )
    step = args.step

    print("Shape before index renaming:", res_df.shape)

    # rename ids
    res_df = rename_ids(gt_df, res_df, step)

    print("Shape after index renaming:", res_df.shape)

    # save
    save_name = DATA_PATH + args.res + "/filtered_trajectories_y0_x0_basic_renamed.csv"
    res_df.to_csv(save_name, index=False)
