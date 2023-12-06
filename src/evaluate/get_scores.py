import jax
import jax.numpy as jnp
import pandas as pd 
import torch
import numpy as np

import matplotlib.pyplot as plt
from IPython import display

import ott
from ott import utils
from ott.math import utils as mu
import tqdm
from ott import problems
from ott.geometry import geometry, pointcloud, costs
from ott.solvers import linear
from ott.solvers.linear import acceleration, sinkhorn, sinkhorn_lr
from ott.tools.sinkhorn_divergence import sinkhorn_divergence
from ott.problems.linear import linear_problem

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import os

from .preprocess_simulated import SimulatedData

class OtEvaluation():
    """
    Simple class for running some OT analyses
    """
    # Class variable for storing ot solution
    ot = None

    def __init__(self, cfg, image_simulated, image_ot_path, results_path):
        self.args = cfg.evaluate
        self.image_ot_path = image_ot_path
        self.results_path = results_path
        self.verbose = cfg.verbose

        # Load simulated data
        self.simulated_data = SimulatedData(cfg, image_simulated, None)
        # self.position_df = pd.read_csv(image_simulated, index_col=0)
        # Load data

    def compute_and_store_scores(self):
        # Iterate through all frames in directory
        # Progress
        if self.verbose:
            print("\n=========================================")
            print("Computing Scores for OT")
            print("=========================================\n")
            print(f'Currently processing:')
        
        # Iterate through all cuts
        scores = []

        # Get ot information type
        dir_name = [dir for dir in os.listdir(self.image_ot_path) if dir.startswith("prob_matrix")][0]
        for file_name in os.listdir(self.image_ot_path / dir_name):
            if file_name.endswith(".npy"):
                # Get frame numbers
                frames = file_name.split(".")[0].split("-")
                curr = self.args.frames[int(frames[0])]
                next = self.args.frames[int(frames[1])]

                if self.verbose:
                    print(f"\nFrames: {curr} - {next}")
                
                # Retrieve ot matrix
                self.ot_matrix = torch.load(self.image_ot_path / dir_name / file_name)
                self.ot_matrix = np.array(self.ot_matrix)
                
                # Extract droplet and embedding features for current frame
                self.x_df = self.simulated_data.get_plain_filtered_position_df(frame=curr, stride=self.args.stride*2)
                self.y_df = self.simulated_data.get_plain_filtered_position_df(frame=next, stride=self.args.stride*2)

                # Get indices of droplets in each frame
                self.x_indices = np.array(self.x_df.index)
                self.y_indices = np.array(self.y_df.index)

                # Compute common indices
                self.common_indices = np.intersect1d(self.x_indices, self.y_indices)

                # Compute scores
                scores_frame = self.get_scores()
                scores_frame["frames"] = f"{curr}-{next}"
                scores.append(scores_frame)

        # Save scores
        scores_df = pd.DataFrame.from_records(scores)
        
        # Compute average of each column
        avg_row = scores_df.mean(numeric_only=True)
        avg_row = [i for i in avg_row] + ["Mean"]

        max_row = scores_df.max(numeric_only=True)
        max_row = [i for i in max_row] + ["Max"]

        scores_df.loc['avg'] = avg_row
        scores_df.loc['max'] = max_row

        scores_df.to_csv(self.results_path / "scores.csv", index=False)

    def _extract_frame(self, df, frame, stride):
        x = jnp.array(df.iloc[::stride,frame])
        y = jnp.array(df.iloc[1::stride,frame])
        pos = jnp.array([x,y]).T
        return pos

    def _extract_frame_df(self, df, frame, stride):
        x = jnp.array(df.iloc[::stride,frame])
        y = jnp.array(df.iloc[1::stride,frame])
        pos = jnp.array([x,y]).T
        index = df.index[::stride]
        index = [i[:-2] for i in index]
        df = pd.DataFrame(pos, index=index, columns=["x", "y"])
        return df

    def _ot_matrix_to_df(ot_matrix, df_x, df_y):
        df = pd.DataFrame(ot_matrix, index=df_x.index, columns=df_y.index)
        return df

    def plot_data(self):
        x_args = {"s": 10, "label": r"source $x$", "marker": "s", "edgecolor": "k"}
        y_args = {"s": 10, "label": r"target $y$", "edgecolor": "k", "alpha": 0.75}
        plt.figure(figsize=(9, 6))
        plt.scatter(self.x[:, 0], self.x[:, 1], **x_args)
        plt.scatter(self.y[:, 0], self.y[:, 1], **y_args)
        plt.legend()
        plt.show()

    def get_scores(self):
        scores = {}
        if self.args.accuracy:
            for k in self.args.accuracy_k:
                accuracy = self.max_k_accuracy(k=k)
                scores[f'accuracy_{k}'] = accuracy

                if self.verbose:
                    print(f'accuracy_{k}: {accuracy}')
         

        if self.args.rank_accuracy:
            for k_rank in self.args.rank_accuracy_k:
                rank_accuracy = self.max_k_rank_accuracy(k_rank=k_rank)
                scores[f'rank_accuracy_{k_rank}'] = rank_accuracy

                if self.verbose:
                    print(f'rank_accuracy_{k_rank}: {rank_accuracy}')
           
        if self.args.auroc_matchable_threshold_stride == "default":
            stride = self.ot_matrix.shape[0]*self.ot_matrix.shape[1]//50000
        else :
            stride = self.args.auroc_matchable_threshold_stride
        if self.args.auroc_matchable:
            auroc_matchable = self.get_roc_auc_score(threshold_stride=stride)
            scores[f'auroc_matchable_{stride}'] = auroc_matchable

            if self.verbose:
                print(f'AUROC Matchable: {auroc_matchable}')

        if self.args.auroc_all_threshold_stride == "default":
            stride = self.ot_matrix.shape[0]*self.ot_matrix.shape[1]//50000
        else :
            stride = self.args.auroc_all_threshold_stride
        if self.args.auroc_all:
            auroc_all = self.get_roc_auc_score_all(threshold_stride=stride)
            scores[f'auroc_all_{stride}'] = auroc_all

            if self.verbose:
                print(f'AUROC All: {auroc_all}')


        if self.args.auroc_rank_matchable:
            auroc_rank_matchable = self.get_roc_auc_score_rank(max_rank=self.args.auroc_rank_matchable_num_ranks)
            scores[f'auroc_rank_matchable_{self.args.auroc_rank_matchable_num_ranks}'] = auroc_rank_matchable

            if self.verbose:
                print(f'AUROC Rank Matchable: {auroc_rank_matchable}')

        if self.args.auroc_rank_all:
            auroc_rank_matchable = self.get_roc_auc_score_rank_all(max_rank=self.args.auroc_rank_all_num_ranks)
            scores[f'auroc_rank_matchable_{self.args.auroc_rank_all_num_ranks}'] = auroc_rank_matchable

            if self.verbose:
                print(f'AUROC Rank Matchable All: {auroc_rank_matchable}')
        
        return scores

    def _get_bin_ot_matrix(self, mat, axis=1):
        if axis==1:
            bin_ot_mat = np.zeros_like(mat)
            bin_ot_mat[np.arange(len(mat)), mat.argmax(1)] = 1
        elif axis==0:
            bin_ot_mat = np.zeros_like(mat)
            bin_ot_mat[mat.argmax(0), np.arange(len(mat[0,:]))] = 1
        elif axis==-1:
            bin_ot_mat = np.zeros_like(mat)
            bin_ot_mat[np.arange(len(mat)), mat.argmax(1)] = 1

            bin_ot_mat2 = np.zeros_like(mat)
            bin_ot_mat2[mat.argmax(0), np.arange(len(mat[0,:]))] = 1
            bin_ot_mat = np.bitwise_and(bin_ot_mat.astype(int), bin_ot_mat2.astype(int))
        
        return bin_ot_mat

    def _get_thresholded_bin_ot_matrix(self, threshold=0.5, max=False):
        
        thresholded_ot_mat = np.array(self.ot_matrix.copy())
        thresholded_ot_mat[thresholded_ot_mat <= threshold] = 0

        if max == True:
            thresholded_ot_mat = self._get_bin_ot_matrix(thresholded_ot_mat, axis=-1)
        else:
            thresholded_ot_mat[self.ot_matrix > threshold] = 1
        
        return thresholded_ot_mat

    def ot_matrix_to_df(self):
        df = pd.DataFrame(self.ot_matrix, index=self.x_df.index, columns=self.y_df.index)
        return df

    def max_k_accuracy(self, k):
        """
        Calculate the accuracy of the OT matrix - only consider the k largest entries
        If k == -1, consider all entries
        If there are ties, consider all tied entries
        """
        max_mat = np.zeros_like(self.ot_matrix)

        # Only keep maximal entries (choices of algorithm)
        max_mat[np.arange(len(self.ot_matrix)), self.ot_matrix.argmax(1)] = self.ot_matrix.max(1)

        sorted_max = np.sort(max_mat.max(1))

        if k > len(sorted_max) or k == -1:
            k = len(sorted_max)
            
        # Identify k largest element
        k_largest_elem = sorted_max[-k]

        # Filter all choices where the value is lower than the k-largest
        max_mat[max_mat < k_largest_elem] = 0

        # Deal with ties
        k_eff = np.count_nonzero(max_mat)

        # Make matrix binary
        max_mat[max_mat > 0] = 1 

        # Get indices of droplets in each frame
        x_indices = np.array(self.x_df.index)
        y_indices = np.array(self.y_df.index)

        # First filter all droplets that do not have a match in the other frame
        common_indices = self.common_indices

        # True labels
        true_labels = self.ot_matrix_to_df().astype(bool)
        true_labels.iloc[:,:]=0
        true_labels.loc[common_indices, common_indices] = np.identity(len(common_indices))

        # Return the normalized score
        score = np.bitwise_and(max_mat.astype(int), true_labels.astype(int)).sum().sum()/(k_eff)

        # # Code for when matrix symm
        # max_mat = np.zeros_like(self.ot_matrix)

        # # Only keep maximal entries (choices of algorithm)
        # max_mat[np.arange(len(self.ot_matrix)), self.ot_matrix.argmax(1)] = self.ot_matrix.max(1)

        # # Identify k largest element
        # k_largest_elem = np.sort(max_mat.max(1))[-k]

        # # Filter all choices where the value is lower than the k-largest
        # max_mat[max_mat < k_largest_elem] = 0

        # # Make matrix binary
        # max_mat[max_mat > 0] = 1 

        # # Return the normalized score
        # score = np.diagonal(max_mat).sum()/k
        return score

    def max_k_rank_accuracy(self, k_rank):
        # Get indices of droplets in each frame
        x_indices = np.array(self.x_df.index)
        y_indices = np.array(self.y_df.index)

        # First filter all droplets that do not have a match in the other frame
        common_indices = self.common_indices

        # Filter indices by dropets which can be matched
        df = self.ot_matrix_to_df()

        # Filter indices by dropets which can be matched
        np_df = np.array(df)
        max_mat = np.zeros_like(np_df)
        
        if k_rank == None or k_rank > len(df.columns):
            k_rank = len(df.columns)
            
        for rank in range(len(df.columns)):
            if rank == k_rank:
                break
            
            # Only keep maximal entries (choices of algorithm)-> Add to set of options
            max_mat[np.arange(len(np_df[:,0])), np_df.argmax(1)] += 1

            # Set these to zero
            np_df[np.arange(len(np_df[:,0])), np_df.argmax(1)] = 0



        # # First filter all droplets that do not have a match in the other frame
        # common_indices = np.intersect1d(x_indices, y_indices)

        # True labels
        true_labels = self.ot_matrix_to_df().astype(bool)
        true_labels.iloc[:,:]=0
        true_labels.loc[common_indices, common_indices] = np.identity(len(common_indices))

        # Return the normalized score
        score = np.bitwise_and(max_mat.astype(int), true_labels.astype(int)).sum().sum()/len(common_indices)

        return score

    def get_roc_auc_score(self, threshold_stride = 1):
        """
        Calculate the ROC-AUC score for the OT matrix
        Filter pairs that are not possible to match
        Use all possible threshold values as thresholds for AUROC
        """
        # Get indices of droplets in each frame
        x_indices = np.array(self.x_df.index)
        y_indices = np.array(self.y_df.index)
        
        # Get all possible threshold values
        threshold_values = np.sort(self.ot_matrix.reshape(-1))
        threshold_values = threshold_values[threshold_values > 0]
        threshold_values = np.unique(threshold_values)
        
        # First filter all droplets that do not have a match in the other frame
        common_indices = self.common_indices

        # Create true labels vector -> 1 if droplet is matched
        true_labels = np.identity(len(common_indices)).flatten()
        y_true = np.concatenate([true_labels]*len(threshold_values[::threshold_stride]))

        # Create predictions vector -> 1 if droplet is matched
        y_score = np.array([])

        # Filter indices by dropets which can be matched
        df_filtered = self.ot_matrix_to_df()

        # Filter all values below threshold
        df_filtered = df_filtered[df_filtered.index.isin(df_filtered.columns)]
        df_filtered = df_filtered.loc[:, df_filtered.index]

        y_score = np.array([])
        np_filtered = np.array(df_filtered)
        
        # for threshold in tqdm.tqdm(threshold_values[::threshold_stride]):
        for threshold in threshold_values[::threshold_stride]:
            np_t = np_filtered.copy()
            np_t[np_t < threshold] = 0
            np_t[np_t > 0] = 1
            y_score = np.concatenate((y_score, np_t.flatten()))

        # Calculate ROC-AUC score
        # return roc_auc_score(y_true, y_score)
        return average_precision_score(y_true, y_score)

    def get_roc_auc_score_all(self, threshold_stride = 1):
        """
        Calculate the ROC-AUC score for the OT matrix
        DO NOT Filter pairs that are not possible to match
        Use all possible threshold values as thresholds for AUROC
        """
        # Get indices of droplets in each frame
        x_indices = np.array(self.x_df.index)
        y_indices = np.array(self.y_df.index)
        
        # Get all possible threshold values
        threshold_values = np.sort(self.ot_matrix.reshape(-1))
        threshold_values = threshold_values[threshold_values > 0]
        threshold_values = np.unique(threshold_values)

        # First filter all droplets that do not have a match in the other frame
        common_indices = self.common_indices

        # Create true labels vector -> 1 if droplet is matched
        true_labels = self.ot_matrix_to_df().astype(bool)
        true_labels.iloc[:,:]=0
        true_labels.loc[common_indices, common_indices] = np.identity(len(common_indices))
        true_labels = np.array(true_labels).flatten()
        y_true = np.concatenate([true_labels]*len(threshold_values[::threshold_stride]))

        # Create predictions vector -> 1 if droplet is matched
        y_score = np.array([])

        # Filter indices by dropets which can be matched
        np_df = np.array(self.ot_matrix)
        y_score = np.array([])
        
        # for threshold in tqdm.tqdm(threshold_values[::threshold_stride]):
        for threshold in threshold_values[::threshold_stride]:
            np_t = np_df.copy()
            np_t[np_t < threshold] = 0
            np_t[np_t > 0] = 1
            y_score = np.concatenate((y_score, np_t.flatten()))

            
        # Calculate ROC-AUC score
        return average_precision_score(y_true, y_score)

    def get_roc_auc_score_rank(self, max_rank = None):
        """
        Calculate the ROC-AUC score for the OT matrix
        DO NOT Filter pairs that are not possible to match
        Use all possible threshold values as thresholds for AUROC
        """
        # Get indices of droplets in each frame
        x_indices = np.array(self.x_df.index)
        y_indices = np.array(self.y_df.index)

        # First filter all droplets that do not have a match in the other frame
        common_indices = self.common_indices
        
        # Filter indices by dropets which can be matched
        df_filtered = self.ot_matrix_to_df()

        # Filter all values below threshold
        df_filtered = df_filtered[df_filtered.index.isin(df_filtered.columns)]
        df_filtered = df_filtered.loc[:, df_filtered.index]

        # Set max rank
        if max_rank == None or max_rank > len(df_filtered.columns):
            max_rank = len(df_filtered.columns)

        # Create true labels vector -> 1 if droplet is matched
        true_labels = np.identity(len(common_indices)).flatten()
        y_true = np.concatenate([true_labels]*max_rank)
        # Create predictions vector -> 1 if droplet is matched
        y_score = np.array([])

        # Filter indices by dropets which can be matched
        np_df = np.array(df_filtered)
        max_mat = np.zeros_like(np_df)

        # for rank in tqdm.tqdm(range(len(df_filtered.columns))):
        for rank in range(len(df_filtered.columns)):
            if rank == max_rank:
                break

            # Only keep maximal entries (choices of algorithm)-> Add to set of options
            max_mat[np.arange(len(np_df[:,0])), np_df.argmax(1)] += 1

            # Set these to zero
            np_df[np.arange(len(np_df[:,0])), np_df.argmax(1)] = 0

            y_score = np.concatenate((y_score, max_mat.flatten()))
            
        # Calculate ROC-AUC score
        return average_precision_score(y_true, y_score)

    def get_roc_auc_score_rank_all(self, max_rank = None):
        """
        Calculate the ROC-AUC score for the OT matrix
        DO NOT Filter pairs that are not possible to match
        Use all possible threshold values as thresholds for AUROC
        """
        # Get indices of droplets in each frame
        x_indices = np.array(self.x_df.index)
        y_indices = np.array(self.y_df.index)

        # First filter all droplets that do not have a match in the other frame
        common_indices = self.common_indices

        # Filter indices by dropets which can be matched
        df = self.ot_matrix_to_df()

        # Set max rank
        if max_rank == None or max_rank > len(df.columns):
            max_rank = len(df.columns)

        # Create true labels vector -> 1 if droplet is matched
        true_labels = self.ot_matrix_to_df().astype(bool)
        true_labels.iloc[:,:]=0
        true_labels.loc[common_indices, common_indices] = np.identity(len(common_indices))
        true_labels = np.array(true_labels).flatten()
        y_true = np.concatenate([true_labels]*max_rank)

        # Create predictions vector -> 1 if droplet is matched
        y_score = np.array([])

        # Filter indices by dropets which can be matched
        np_df = np.array(df)
        y_score = np.array([])
        max_mat = np.zeros_like(np_df)
        
        # for rank in tqdm.tqdm(range(len(df.columns))):
        for rank in range(len(df.columns)):
            if rank == max_rank:
                break
            
            # Only keep maximal entries (choices of algorithm)-> Add to set of options
            max_mat[np.arange(len(np_df[:,0])), np_df.argmax(1)] += 1

            # Set these to zero
            np_df[np.arange(len(np_df[:,0])), np_df.argmax(1)] = 0

            y_score = np.concatenate((y_score, max_mat.flatten()))

            
        # Calculate ROC-AUC score
        return average_precision_score(y_true, y_score)
        
    def get_bin_accuracy(self):
        return self.best_k_score(self.ot_matrix, k=len(np.diagonal(self.ot_matrix)))