from enum import Enum

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from tqdm import tqdm
import torch

from ott import utils
from ott.math import utils as mu
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from pathlib import Path
import os

EmbDist = Enum('EmbdDist', ['euclid', 'cosine'])


@jax.tree_util.register_pytree_node_class
class SpatioVisualCost(costs.CostFn):
    """Cost function for combined features (position and visual embedding information)."""

    def __init__(self, alpha, beta=None, gamma=None, use_nr_cells=True, embedding_dist: EmbDist=EmbDist.euclid):
        super().__init__()
        self.alpha = alpha
        if (beta is None) or (beta == 'None'):
            self.beta = 1 - alpha
        else:
            self.beta = beta

        if (gamma is None) or (gamma == 'None'):
            self.gamma = 0.5 * self.alpha
        else:
            self.gamma = gamma

        self.use_nr_cells = use_nr_cells
        self.embedding_dist = embedding_dist

    def pairwise(self, x, y):
        if x.size <= 2:
            return self.pairwise_spatial(x, y)
        elif self.use_nr_cells:
            return self.alpha * self.pairwise_spatial(x, y) + self.beta * self.pairwise_embedding(x, y) + self.gamma * self.pairwise_nr_cells(x, y)
        else:
            return self.alpha * self.pairwise_spatial(x, y) + self.beta * self.pairwise_embedding(x, y)

    def pairwise_spatial(self, x, y):
        return mu.norm(x[:2] - y[:2])
    
    def pairwise_nr_cells(self, x, y):
        return mu.norm(x[2] - y[2])

    def pairwise_embedding(self, x, y):
        if self.embedding_dist == EmbDist.euclid:
            return mu.norm(x[3:] - y[3:])
        elif self.embedding_dist == EmbDist.cosine:
            return 1 - jnp.dot(x[3:], y[3:]) / (jnp.linalg.norm(x[3:]) * jnp.linalg.norm(y[3:]))
        else:
            raise NotImplementedError(f'Distance {self.embedding_dist} not implemented. In ot.py.')

    # The two functions below are necessary, because ott instantiates more classes of this type under the hood.
    # If we don't define these 2 functions, the newly created objects will have default parameters.
    def tree_flatten(self):  # noqa: D102
        return (), (self.alpha, self.beta, self.gamma, self.use_nr_cells, self.embedding_dist)

    @classmethod
    def tree_unflatten(cls, aux_data, children):  # noqa: D102
        del children
        return cls(*aux_data)


class OptimalTransport:
    """Class for computing optimal transport between two point clouds."""

    def __init__(self, cfg):
        # Get OT config parameters
        self.args = cfg.track
        self.verbose = cfg.verbose
        self.tqdm_disable = cfg.tqdm_disable

        # Instantiate cost function and params
        self.embedding_dist = EmbDist[self.args.embedding_dist]
        self.cost_fn = SpatioVisualCost(alpha=self.args.alpha,
                                        beta=self.args.beta,
                                        gamma=self.args.gamma,
                                        use_nr_cells=self.args.use_nr_cells,
                                        embedding_dist=self.embedding_dist)

        # Instantiate Sinkhorn solver only once for better performance
        self.solver = jax.jit(
            sinkhorn.Sinkhorn(
                max_iterations=self.args.max_iterations
            )
        )

    def compute_ot_matrix(self, features_curr, features_next):
        """Compute optimal transport between two point clouds."""
        # THIS IS NECESSARY - PYTHON IS PASS BY REFERENCE
        x = features_curr.copy()
        y = features_next.copy()

        # Scale the data, so that cost weights and epsilon are meaningful
        spatial_dists = euclidean_distances(x[:, :2], y[:, :2])
        if x.shape[1] > 3 and self.embedding_dist == EmbDist.cosine:
            visual_dists = cosine_distances(x[:, 3:], y[:, 3:]) 
        elif x.shape[1] > 3 and self.embedding_dist == EmbDist.euclid:
            visual_dists = euclidean_distances(x[:, 3:], y[:, 3:])
        else:
            visual_dists = 0

        # Use quantiles to avoid outliers
        spatial_dist_range = np.quantile(spatial_dists, 0.95)
        if self.verbose:
            print("Distances before scaling")
            print(f"Spatial: 95%-quantile: {spatial_dist_range:.3f}, max: {np.max(spatial_dists):.3f}")
            print(f"Visual embeddings: 95%-quantile: {np.quantile(visual_dists, 0.95):.3f}, max: {np.max(visual_dists):.3f}")

        visual_dist_range = 1.0
        if self.embedding_dist == EmbDist.cosine:
            # Cosine distance is bounded by definition
            visual_dist_range = 2.0
        elif self.embedding_dist == EmbDist.euclid and np.sum(visual_dists) > 0:
            # We don't want to scale by visual_dist_range if it's zero, 
            # which can sometimes happen if all droplets are empty.
            visual_dist_range = np.quantile(visual_dists, 0.95)

        x[:, :2] = visual_dist_range * x[:, :2] / spatial_dist_range
        y[:, :2] = visual_dist_range * y[:, :2] / spatial_dist_range

        if self.verbose:
            new_spatial_dists = euclidean_distances(x[:, :2], y[:, :2])
            print("Distances after scaling")
            print(f"Spatial: 95%-quantile: {np.quantile(new_spatial_dists, 0.95):.3f}, max: {np.max(new_spatial_dists):.3f}")
            print(f"Visual embeddings: 95%-quantile: {np.quantile(visual_dists, 0.95):.3f}, max: {np.max(visual_dists):.3f}")

        # Create geometry and problem
        if self.args.relative_epsilon == "default":
            geom = pointcloud.PointCloud(jnp.array(x), jnp.array(y), cost_fn=self.cost_fn)
        else:
            # We are using relative epsilon, because the ratio between mean values in the cost matrix and the epsilon
            # is more informative than the epsilon itself.
            geom = pointcloud.PointCloud(jnp.array(x), jnp.array(y), epsilon=self.args.relative_epsilon,
                                         relative_epsilon=True, cost_fn=self.cost_fn)

        ot_prob = linear_problem.LinearProblem(geom,
                                               tau_a=self.args.tau_a,
                                               tau_b=self.args.tau_b)
        
        # Solve problem using given solver
        ot = jax.jit(self.solver)(ot_prob)

        if self.args.print_convergence:
            # TODO: Think what to do when we gat a nan in the first step - it happens for one testing configuration.
            print(f'Sinkhorn has converged: {ot.converged}, in {jnp.sum(ot.errors > -1)} iterations\n'
                  f'Error upon last iteration: {ot.errors[(ot.errors > -1)][-1]:.4e}\n')

        return ot.matrix

    def extract_features(self, droplet_df, embedding_df):
        """Extract features from droplet and embedding dataframes."""
        # Get droplet features
        droplet_df = droplet_df[['center_x', 'center_y', 'nr_cells', 'droplet_id']]
        droplet_df = droplet_df.set_index(['droplet_id'])

        # Get embedding features
        embedding_df = embedding_df[['embeddings', 'droplet_id']]
        embedding_df = embedding_df.set_index(['droplet_id'])

        # Merge droplet and embedding features
        features_df = droplet_df.join(embedding_df, how='inner')
        features_df = features_df.reset_index()

        # Concatenate features
        features_df['combined'] = features_df.apply(
            lambda row: np.concatenate(([row['center_x'], row['center_y'], row['nr_cells']], row['embeddings']), axis=0), axis=1)
        
        # Stack 'combined' into a 2D numpy array and convert to float32
        features = np.stack(features_df['combined'].values).astype(np.float32)

        return features

    def compute_and_store_ot_matrices_cut(self, droplet_df, embedding_df, cut_ot_path, max_frame):
        """Compute and store optimal transport matrices for a single cut."""

        # Iterate through frames

        for i in tqdm(range(max_frame), disable=self.tqdm_disable):
            # Extract droplet and embedding features for current frame
            if (i == 0):
                droplet_df_curr = droplet_df[droplet_df['frame'] == i]
                embedding_df_curr = embedding_df[embedding_df['frame'] == i]
                features_curr = self.extract_features(droplet_df_curr, embedding_df_curr)
            else:
                features_curr = features_next.copy()

            droplet_df_next = droplet_df[droplet_df['frame'] == i + 1]
            embedding_df_next = embedding_df[embedding_df['frame'] == i + 1]
            features_next = self.extract_features(droplet_df_next, embedding_df_next)

            # Compute optimal transport matrix
            ot_matrix = self.compute_ot_matrix(features_curr, features_next)

            # Save matrix
            filename = f'{i}-{i + 1}.npy'
            torch.save(ot_matrix, cut_ot_path / filename)

    def compute_and_store_ot_matrices_all(self, image_feature_path, image_ot_path):
        # Progress
        if self.verbose:
            print("\n=========================================")
            print("Computing OT Matrices For all Cuts")
            print("=========================================\n")
            print(f'Currently computing ot matrices for cut:')

        # Iterate through all cuts
        num_files = len(os.listdir(image_feature_path))
        for file_name in os.listdir(image_feature_path):
            # Use droplets file for name reference only - all pairs (droplet, embeddings) must exist 
            if not file_name.startswith("droplets"):
                continue

            # Progress
            if self.verbose:
                print(file_name)

            # Get file names for current cut
            cut_feature_droplet_name = file_name
            cut_feature_embedding_name = file_name.replace("droplets", "embeddings")[:-4] + ".npy"

            cut_ot_name = file_name.replace("droplets", "ot_matrix")
            cut_ot_path = Path(image_ot_path / cut_ot_name[:-4])

            if not os.path.exists(cut_ot_path):
                os.makedirs(cut_ot_path)

            # Get features of current cut
            droplet_df = pd.read_csv(image_feature_path / cut_feature_droplet_name)
            embedding_df = np.load(image_feature_path / cut_feature_embedding_name, allow_pickle=True).item()
            embedding_df = pd.DataFrame.from_dict(embedding_df)

            # Save embedding df
            if self.args.save_embedding_df:
                name = "df_" + cut_feature_embedding_name.replace(".npy", ".csv")
                embedding_df.to_csv(image_feature_path / name, index=False)

            max_frame = droplet_df['frame'].max()

            # Compute ot matrices for current cut
            self.compute_and_store_ot_matrices_cut(droplet_df, embedding_df, cut_ot_path, max_frame)
