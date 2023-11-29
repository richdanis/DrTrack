import time
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import tqdm

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
        self.args = cfg.track.unb_reg_sinkhorn
        self.verbose = cfg.verbose

        # Instantiate cost function
        self.cost_fn = SpatioVisualCost(alpha=self.args.alpha, beta=self.args.beta)

        # Instantiate solver for fused_gw version for better performance
        if self.args.ot_type == "fused_gw":
            self.solver = jax.jit(
                sinkhorn.Sinkhorn(
                    max_iterations=self.args.max_iterations
                )
            )  

    def get_ot_matrix(self, x, y):
        """Compute optimal transport between two point clouds."""
        # Create geometry and problem
        geom = pointcloud.PointCloud(x, y, epsilon=self.args.epsilon)
        ot_prob = linear_problem.LinearProblem(geom,
                                                tau_a = self.args.tau_a,
                                                tau_b = self.args.tau_b)
        
        # Solve problem using given solver
        ot = jax.jit(self.solver)(ot_prob)

        if self.verbose:
            print(f'Sinkhorn has converged: {ot.converged}, in {jnp.sum(ot.errors > -1)} iterations\n'
                f'Error upon last iteration: {ot.errors[(ot.errors > -1)][-1]:.4e}')
            
        return ot.matrix
        
    def get_ot_matrices_all_frames(self, cfg, cut_feature_path):
        # Iterate through all pairs of frames 
        ot_matrices = np.array()

        # Get features
        features = np.load(cut_feature_path)

        for i in tqdm(range(len(cfg.track.frame_range)-1), disable=not cfg.verbose):
            # Get frame numbers
            frame_curr = features[i,:,:]
            frame_next = features[i+1,:,:]

            # # Get point clouds
            # x = np.load(feature_path / f'frame_{frame1}_features.npy')
            # y = np.load(feature_path / f'frame_{frame2}_features.npy')

            # Compute optimal transport matrix
            ot_matrix = self.get_ot_matrix(frame_curr, frame_next)

            # Save matrix
            ot_matrices.append(ot_matrix)
            


        
        return 