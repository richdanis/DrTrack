import time
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch 

import matplotlib.pyplot as plt
from IPython import display

from ott import utils
from ott.math import utils as mu
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from pathlib import Path
import os

@jax.tree_util.register_pytree_node_class
class SpatioVisualCost(costs.CostFn):
    """Cost function for combined features (position and visual embedding information)."""
    def __init__(self, alpha=0.5, beta=0.5):
        self.alpha=alpha
        self.beta=beta

    def pairwise(self, x, y):
        if x.ndim<=2:
            return self.pairwise_spatial(x,y)
        else:
            return self.beta*self.pairwise_embedding(x,y)+self.alpha*self.pairwise_spatial(x,y)
    
    def pairwise_spatial(self, x, y):
        return mu.norm(x[:2] - y[:2])
    
    def pairwise_embedding(self, x, y):
        return mu.norm(x[2:] - y[2:])

class OptimalTransport:
    """Class for computing optimal transport between two point clouds."""
    def __init__(self, cfg):
        # Get OT config parameters
        self.args = cfg.track
        self.verbose = cfg.verbose
        self.tqdm_disable = cfg.tqdm_disable

        # Instantiate cost function
        self.cost_fn = SpatioVisualCost(alpha=self.args.alpha, beta=self.args.beta)

        # Instantiate solver for fused_gw version for better performance
        self.solver = jax.jit(
            sinkhorn.Sinkhorn(
                max_iterations=self.args.max_iterations
            )
        )
            

    def compute_ot_matrix(self, x, y):
        """Compute optimal transport between two point clouds."""
        # Create geometry and problem
        geom = pointcloud.PointCloud(x, y, epsilon=self.args.epsilon)
        ot_prob = linear_problem.LinearProblem(geom,
                                                tau_a = self.args.tau_a,
                                                tau_b = self.args.tau_b)
        
        # Solve problem using given solver
        ot = jax.jit(self.solver)(ot_prob)

        # if self.verbose:
        #     print(f'Sinkhorn has converged: {ot.converged}, in {jnp.sum(ot.errors > -1)} iterations\n'
        #         f'Error upon last iteration: {ot.errors[(ot.errors > -1)][-1]:.4e}')
            
        return ot.matrix
        
    def compute_and_store_ot_matrices_cut(self, features, cut_ot_path):
        # Iterate through all pairs of frames 
        num_frames = features.shape[0]

        for i in tqdm(range(num_frames-1), self.tqdm_disable):
            # Get frame numbers
            frame_curr = features[i,:,:]
            frame_next = features[i+1,:,:]

            # Compute optimal transport matrix
            ot_matrix = self.compute_ot_matrix(frame_curr, frame_next)

            # Save matrix
            filename = f'{i}-{i+1}.npy'
            torch.save(ot_matrix, cut_ot_path / filename)
    
    def compute_and_store_ot_matrices_all(self, image_feature_path, image_ot_path, features):
        # Progress
        if self.verbose:
            print("\n=========================================")
            print("Computing OT Matrices For all Cuts")
            print("=========================================\n")
            print(f'Currently computing ot matrices for cut:')
        
        # Iterate through all cuts
        for file_name in os.listdir(image_feature_path):
            # Progress
            if self.verbose:
                print(file_name)


            embeddings = np.load(image_feature_path /'embeddings_matched_data_y0_x0.npy.npy', allow_pickle=True).item()
            embeddings_df = pd.DataFrame(embeddings)
            print(embeddings_df.head())

            # TODO
            if not file_name.startswith("droplets_"):
                continue

            # Get cut name
            cut_name = file_name.replace("droplets_", "ot_matrix_")
            cut_ot_path = Path(image_ot_path / cut_name)

            if not os.path.exists(cut_ot_path):
                os.makedirs(cut_ot_path)

            # Get features of current cut
            ## TODO: 
            #features = np.load(image_feature_path / cut / 'features.npy')

            # Compute and store ot matrices for current cut
            self.compute_and_store_ot_matrices_cut(features, cut_ot_path)