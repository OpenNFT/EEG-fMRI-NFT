import numpy as np
import argparse
import pandas as pd
import math
import os
import glob
import pickle
import json
import joblib  # Added missing import
from itertools import combinations
from pathlib import Path
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram
import nibabel as nib
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS, TSNE
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm.autonotebook import tqdm

# ============================================================================
# CLUSTERING FUNCTIONS
# ============================================================================

def load_csv_data_for_clustering(csv_path, channel_name=None):
    """Load your CSV data and prepare it for clustering."""
    
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} rows, {len(df['subject_id'].unique())} subjects")
    
    # Filter for specific channel if requested
    if channel_name is not None:
        df = df[df['channel_name'] == channel_name].copy()
        
        if len(df) == 0:
            print(f"ERROR: No data found for channel {channel_name}")
            available_channels = sorted(pd.read_csv(csv_path)['channel_name'].unique())
            print(f"Available channels: {available_channels}")
            return None, None, None
    
    # Performance metrics for clustering
    performance_columns = [
        'pearson r test', 'pearson p test', 'MSE test', 'nMSE test', 'r2 test',
        'MSE val', 'r2 val', 'pearson val'
    ]
    
    available_columns = [col for col in performance_columns if col in df.columns]
    
    if channel_name is None:
        # Multi-channel: create pivot table (subjects x channels)
        pivot_data = df.pivot_table(
            index='subject_id', 
            columns='channel_name', 
            values='pearson r test', 
            aggfunc='mean'
        )
        
        pivot_data = pivot_data.fillna(0)
        subjects = pivot_data.index.tolist()
        data_matrix = pivot_data.values
        feature_names = [f"{ch}_pearson_r" for ch in pivot_data.columns]
        
    else:
        # Single channel: use all performance metrics
        df_sorted = df.sort_values('subject_id')
        subjects = df_sorted['subject_id'].tolist()
        data_matrix = df_sorted[available_columns].values
        feature_names = available_columns
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        data_matrix = imputer.fit_transform(data_matrix)
    
    print(f"Loaded {len(subjects)} subjects with {len(feature_names)} features")
    
    return data_matrix, subjects, feature_names

def create_distance_matrix(data_matrix, metric='cosine'):
    """Create subject×subject distance matrix."""
    distances = pdist(data_matrix, metric=metric)
    distance_matrix = squareform(distances)
    return distance_matrix

def calculate_wcss(distance_matrix, labels, n_clusters):
    """Compute Within-Cluster Sum of Squares from distance matrix."""
    wcss = 0
    
    for i in range(n_clusters):
        cluster_points = np.where(labels == i)[0]
        if len(cluster_points) > 1:
            # Get pairwise distances within cluster
            intra_cluster_distances = distance_matrix[np.ix_(cluster_points, cluster_points)]
            wcss += np.sum(intra_cluster_distances) / (2 * len(cluster_points))
    
    return wcss

def projection_plot_clusters(distance_matrix, clusters, subjects, channel_name=None, special_markers=(), show_plot=False):
    """Create 2D projection plot of clusters using MDS or t-SNE."""
    
    if not show_plot:
        return  # Skip plotting
        
    # Choose projection method
    if len(distance_matrix.shape) == 2 and distance_matrix.shape[1] == distance_matrix.shape[0]:
        explainer = MDS(n_components=2, dissimilarity='precomputed', random_state=93)
        coordinates = explainer.fit_transform(distance_matrix)
    else:
        explainer = TSNE(n_components=2, random_state=93)
        coordinates = explainer.fit_transform(distance_matrix)

    # Create colors for clusters
    colors = cm.get_cmap('tab20', np.max(clusters) + 1)
    point_colors = [colors(cluster) for cluster in clusters]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=point_colors, s=100, alpha=0.7, edgecolors='black')
    
    # Add subject labels
    for i, (x, y) in enumerate(coordinates):
        plt.annotate(subjects[i], (x, y), xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.8)
    
    # Mark special subjects if specified
    if special_markers:
        extra_markers_x = []
        extra_markers_y = []
        for i, subject in enumerate(subjects):
            if subject in special_markers:
                extra_markers_x.append(coordinates[i, 0])
                extra_markers_y.append(coordinates[i, 1])
        if extra_markers_x:
            plt.scatter(extra_markers_x, extra_markers_y, marker='*', 
                       color='red', s=200, edgecolors='black', label='Selected subjects')
            plt.legend()
    
    # Title and labels
    title = f'Subject Clustering Projection'
    if channel_name:
        title += f' ({channel_name})'
    plt.title(title, fontsize=16)
    plt.xlabel('Dimension 1', fontsize=14)
    plt.ylabel('Dimension 2', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add cluster information
    unique_clusters = np.unique(clusters)
    cluster_info = f"Clusters: {len(unique_clusters)}\n"
    for cluster_id in unique_clusters:
        count = np.sum(clusters == cluster_id)
        cluster_info += f"Cluster {cluster_id}: {count} subjects\n"
    
    plt.text(0.02, 0.98, cluster_info, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def agglomerative_clustering_csv(n_clusters, data_matrix, subjects, 
                                subject_labels=None, linkage='average', 
                                metric='cosine', plot=False, channel_name=None,
                                special_markers=()):
    """Perform agglomerative clustering on your CSV data."""
    
    if subject_labels is None:
        subject_labels = subjects
    
    # Create distance matrix
    distance_matrix = create_distance_matrix(data_matrix, metric=metric)
    
    # Perform clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        compute_full_tree=True,
        compute_distances=True,
        linkage=linkage,
    )
    
    clusters = clustering.fit_predict(distance_matrix)
    
    # Calculate metrics
    wcss = calculate_wcss(distance_matrix, clusters, n_clusters)
    
    if n_clusters > 1:
        sil_score = silhouette_score(distance_matrix, clusters, metric='precomputed')
        print(f"Silhouette Score: {sil_score:.3f}")
    
    # Print cluster assignments
    unique_clusters = np.unique(clusters)
    for cluster_id in unique_clusters:
        subjects_in_cluster = [subjects[i] for i in range(len(subjects)) 
                              if clusters[i] == cluster_id]
        print(f"Cluster {cluster_id}: {subjects_in_cluster}")
    
    # Create plots (disabled by default)
    if plot:
        projection_plot_clusters(distance_matrix, clusters, subjects, 
                                channel_name=channel_name, special_markers=special_markers, show_plot=True)
    
    return wcss, clustering, clusters, distance_matrix

def find_optimal_clusters(data_matrix, subjects, max_clusters=10, 
                         linkage='average', metric='cosine', channel_name=None, show_plots=False):
    """Find optimal number of clusters using silhouette score."""
    
    max_clusters = min(max_clusters, len(subjects) - 1)
    wcss_scores = []
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)
    
    # Create distance matrix once
    distance_matrix = create_distance_matrix(data_matrix, metric=metric)
    
    for n_clusters in cluster_range:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage=linkage,
        )
        
        clusters = clustering.fit_predict(distance_matrix)
        
        # Calculate metrics
        wcss = calculate_wcss(distance_matrix, clusters, n_clusters)
        sil_score = silhouette_score(distance_matrix, clusters, metric='precomputed')
        
        wcss_scores.append(wcss)
        silhouette_scores.append(sil_score)
    
    # Find optimal number of clusters
    optimal_silhouette = cluster_range[np.argmax(silhouette_scores)]
    print(f"Optimal clusters (silhouette): {optimal_silhouette}")
    
    return {
        'cluster_range': cluster_range,
        'wcss_scores': wcss_scores,
        'silhouette_scores': silhouette_scores,
        'optimal_silhouette': optimal_silhouette,
    }

# ============================================================================
# MAX-MIN SELECTION FUNCTIONS
# ============================================================================

def greedy_max_min_selection(distance_matrix, n):
    """Greedy algorithm to select n samples that minimize the maximum pairwise distance."""
    num_samples = distance_matrix.shape[0]
    
    # Step 1: Initialize with the pair having the smallest distance
    temp_matrix = distance_matrix + np.eye(num_samples) * np.inf
    best_pair = np.unravel_index(np.argmin(temp_matrix), temp_matrix.shape)
    selected_indices = list(best_pair)
    
    # Step 2: Iteratively add samples
    while len(selected_indices) < n:
        remaining_indices = [i for i in range(num_samples) if i not in selected_indices]
        if len(remaining_indices) == 0:
            break
            
        best_candidate = None
        min_max_distance = np.inf

        for candidate in remaining_indices:
            new_subset = selected_indices + [candidate]
            max_distance = np.max(distance_matrix[np.ix_(new_subset, new_subset)])
            if max_distance < min_max_distance:
                min_max_distance = max_distance
                best_candidate = candidate

        if best_candidate is not None:
            selected_indices.append(best_candidate)
    
    # Calculate final max distance
    final_max_distance = np.max(distance_matrix[np.ix_(selected_indices, selected_indices)])
    return selected_indices, final_max_distance

def exact_max_min_selection(distance_matrix, n, disable_bar=True):
    """Exact algorithm to find optimal n samples that minimize maximum pairwise distance."""
    best_combination = None
    min_max_distance = np.inf
    total_iters = math.comb(distance_matrix.shape[0], n)

    for comb in tqdm(combinations(range(distance_matrix.shape[0]), n), 
                     total=total_iters, disable=disable_bar):
        comb_distances = distance_matrix[np.ix_(comb, comb)]
        max_distance = np.max(comb_distances)
        if max_distance < min_max_distance:
            min_max_distance = max_distance
            best_combination = comb

    return list(best_combination), min_max_distance

def max_min_selection(distance_matrix, n, max_iter=2000000):
    """Choose between exact and greedy max-min selection based on computational complexity."""
    total_iters = math.comb(distance_matrix.shape[0], n)
    if total_iters > max_iter:
        print(f"Too many iterations ({total_iters}), using greedy algorithm")
        return greedy_max_min_selection(distance_matrix, n)
    else:
        return exact_max_min_selection(distance_matrix, n)

def get_closest_n_elements_from_clusters(
    n, distance_matrix, cluster_labels, subjects, 
    plot=False, max_iter=2000000, channel_name=None, 
    target_cluster=None
):
    """Select n closest elements from the largest cluster (or specified cluster)."""
    
    # Determine which cluster to use
    if target_cluster is None:
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        largest_cluster_label = unique_labels[np.argmax(counts)]
        print(f"Using largest cluster (Cluster {largest_cluster_label}) with {np.max(counts)} subjects")
    else:
        largest_cluster_label = target_cluster
        cluster_size = np.sum(cluster_labels == target_cluster)
        print(f"Using specified cluster (Cluster {target_cluster}) with {cluster_size} subjects")
    
    # Get indices of subjects in the target cluster
    indices_largest_cluster = np.where(cluster_labels == largest_cluster_label)[0]
    
    if len(indices_largest_cluster) < n:
        raise ValueError(f"Number of elements to select ({n}) is larger than the number of elements "
                        f"in cluster {largest_cluster_label} ({len(indices_largest_cluster)})")

    # Extract distance submatrix for the target cluster
    cluster_distance_matrix = distance_matrix[np.ix_(indices_largest_cluster, indices_largest_cluster)]
    
    # Perform max-min selection within the cluster
    selected_samples_in_cluster, max_min_distance = max_min_selection(
        cluster_distance_matrix, n, max_iter
    )
    
    # Map back to original indices
    selected_samples = indices_largest_cluster[selected_samples_in_cluster]
    selected_subjects = [subjects[k] for k in selected_samples]
    
    print(f"Selected {len(selected_subjects)} subjects from cluster {largest_cluster_label}:")
    print(f"Selected subjects: {selected_subjects}")
    print(f"Max-min distance: {max_min_distance:.4f}")
    
    if plot:
        projection_plot_clusters(
            distance_matrix, cluster_labels, subjects, 
            channel_name=channel_name, special_markers=selected_subjects, show_plot=True
        )

    return selected_samples, max_min_distance, selected_subjects

def enhanced_clustering_with_selection(
    csv_path, channel_name=None, n_clusters=None, n_to_select=10,
    linkage='average', metric='cosine', max_iterations=2000000,
    output_dir=None, target_cluster=None
):
    """Complete pipeline that combines clustering with intelligent subject selection."""
    
    # Load data
    data_matrix, subjects, feature_names = load_csv_data_for_clustering(
        csv_path, channel_name=channel_name
    )
    
    if data_matrix is None:
        return None
    
    # Create distance matrix
    distance_matrix = create_distance_matrix(data_matrix, metric=metric)
    
    # Determine optimal number of clusters if not specified
    if n_clusters is None:
        optimization_results = find_optimal_clusters(
            data_matrix, subjects, max_clusters=min(10, len(subjects)-1),
            linkage=linkage, metric=metric, channel_name=channel_name
        )
        n_clusters = optimization_results['optimal_silhouette']
        print(f"Auto-determined optimal clusters: {n_clusters}")
    else:
        optimization_results = None
    
    # Perform clustering
    wcss, clustering_model, cluster_labels, _ = agglomerative_clustering_csv(
        n_clusters=n_clusters,
        data_matrix=data_matrix,
        subjects=subjects,
        linkage=linkage,
        metric=metric,
        plot=True,
        channel_name=channel_name
    )
    
    # Select representative subjects from cluster
    done = False
    selected_subjects = []
    max_min_distance = 0
    
    while n_to_select > 0 and not done:
        try:
            selected_samples, max_min_distance, selected_subjects = get_closest_n_elements_from_clusters(
                n_to_select, distance_matrix, cluster_labels, subjects,
                max_iter=max_iterations, plot=True, 
                channel_name=channel_name, target_cluster=target_cluster
            )
            done = True
        except ValueError as e:
            print(f"Warning: {e}")
            n_to_select = n_to_select - 1
            if n_to_select == 0:
                print("Could not select any subjects - cluster too small")
                break
    
    results = {
        'wcss': wcss,
        'n_clusters': n_clusters,
        'cluster_labels': cluster_labels,
        'clustering_model': clustering_model,
        'distance_matrix': distance_matrix,
        'n_selected': len(selected_subjects) if selected_subjects else 0,
        'selected_subjects': selected_subjects,
        'selected_indices': selected_samples if 'selected_samples' in locals() else [],
        'max_min_distance': max_min_distance,
        'subjects': subjects,
        'optimization_results': optimization_results
    }
    
    # Save results if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        channel_prefix = channel_name if channel_name else 'all_channels'
        
        # Save cluster assignments
        cluster_df = pd.DataFrame({
            'subject_id': subjects,
            'cluster': cluster_labels
        })
        cluster_df.to_csv(os.path.join(output_dir, f"{channel_prefix}_clusters.csv"), index=False)
        
        # Save selected subjects
        if selected_subjects:
            selected_df = pd.DataFrame({
                'selected_subject_id': selected_subjects,
                'selection_order': range(len(selected_subjects))
            })
            selected_df.to_csv(os.path.join(output_dir, f"{channel_prefix}_selected_subjects.csv"), index=False)
        
        # Save distance matrix
        distance_df = pd.DataFrame(
            distance_matrix, 
            index=subjects, 
            columns=subjects
        )
        distance_df.to_csv(os.path.join(output_dir, f"{channel_prefix}_distance_matrix.csv"))
        
        print(f"Results saved to: {output_dir}")
    
    return results

# ============================================================================
# LEAVE-ONE-OUT CLUSTER TRAINING
# ============================================================================

def leave_one_out_cluster_training(
        cluster_data,  # Dictionary of {subject_id: (data, target, metadata)}
        selected_subjects,  # List of selected subjects from clustering
        n_channels,
        n_bands,
        n_timepoints,
        channel_names=None,
        inner_k=5,
        regularization_grid_search=10,
        range_channels=None,
        metric='MSE',
        verbose=False,
        estimator=None
):
    """Leave-One-Out training on clustered subjects: Train on N-1, test on 1."""
    
    print(f"\n{'='*80}")
    print("LEAVE-ONE-OUT CLUSTER TRAINING")
    print(f"{'='*80}")
    print(f"Selected subjects: {selected_subjects}")
    print(f"Will perform {len(selected_subjects)} LOO iterations")
    
    if estimator is None:
        estimator = Ridge
    if range_channels is None:
        range_channels = range(n_channels)
    
    all_loo_results = []
    
    # Leave-One-Out loop
    for i, test_subject in enumerate(selected_subjects):
        print(f"\n{'='*60}")
        print(f"LOO Iteration {i+1}/{len(selected_subjects)}")
        print(f"Test Subject: {test_subject}")
        print(f"{'='*60}")
        
        # Get training subjects (all except the test subject)
        train_subjects = [s for s in selected_subjects if s != test_subject]
        print(f"Training subjects: {train_subjects}")
        
        # Combine training data
        train_data_list = []
        train_targets_list = []
        
        for train_subj in train_subjects:
            if train_subj in cluster_data:
                data, target, metadata = cluster_data[train_subj]
                train_data_list.append(data)
                train_targets_list.append(target)
        
        # Concatenate training data
        combined_train_data = np.concatenate(train_data_list, axis=0)
        combined_train_targets = np.concatenate(train_targets_list, axis=0)
        
        # Get test data
        test_data, test_target, test_metadata = cluster_data[test_subject]
        
        # Train models on combined training data
        inner_splitter = KFold(n_splits=inner_k, shuffle=False)
        
        for channel in range_channels:
            channel_name = channel_names[channel] if channel_names and channel < len(channel_names) else f"Channel_{channel}"
            
            # Get training data for this channel
            features_per_channel = n_bands * n_timepoints
            start_idx = channel * features_per_channel
            end_idx = (channel + 1) * features_per_channel
            
            train_ch_data = combined_train_data[:, start_idx:end_idx]
            test_ch_data = test_data[:, start_idx:end_idx]
            
            # Grid search with cross-validation on training data
            if isinstance(regularization_grid_search, int):
                alpha_grid = {'alpha': np.logspace(-6, 6, num=regularization_grid_search)}
            else:
                alpha_grid = regularization_grid_search
                
            grid_search = GridSearchCV(
                estimator=estimator(),
                param_grid=alpha_grid,
                cv=inner_splitter,
                scoring={
                    'pearson': make_scorer(pearson_scorer),
                    'MSE': make_scorer(mean_squared_error, greater_is_better=False),
                    'r2': make_scorer(r2_score)
                },
                refit=metric,
                n_jobs=-1
            )
            
            # Fit on training data
            grid_search.fit(train_ch_data, combined_train_targets)
            
            # Test on held-out subject
            test_predictions = grid_search.predict(test_ch_data)
            
            # Calculate test metrics
            try:
                pearson_r, pearson_p = pearsonr(test_target, test_predictions)
                if np.isnan(pearson_r):
                    pearson_r, pearson_p = 0.0, 1.0
            except:
                pearson_r, pearson_p = 0.0, 1.0
                
            mse = mean_squared_error(test_target, test_predictions)
            r2 = r2_score(test_target, test_predictions)
            
            # Cross-validation metrics on training data
            cv_pearson = np.mean(grid_search.cv_results_['mean_test_pearson'])
            cv_mse = -np.mean(grid_search.cv_results_['mean_test_MSE'])
            cv_r2 = np.mean(grid_search.cv_results_['mean_test_r2'])
            
            # Store results
            loo_result = {
                'loo_iteration': i + 1,
                'test_subject': test_subject,
                'train_subjects': train_subjects,
                'channel': channel,
                'channel_name': channel_name,
                'test_pearson_r': pearson_r,
                'test_pearson_p': pearson_p,
                'test_mse': mse,
                'test_r2': r2,
                'train_pearson_r': cv_pearson,
                'train_mse': cv_mse,
                'train_r2': cv_r2,
                'best_alpha': grid_search.best_params_['alpha'],
            }
            
            all_loo_results.append(loo_result)
        
        # Print summary for this LOO iteration
        iteration_results = [r for r in all_loo_results if r['test_subject'] == test_subject]
        test_pearson_values = [r['test_pearson_r'] for r in iteration_results if not np.isnan(r['test_pearson_r'])]
        
        if test_pearson_values:
            print(f"Test performance - Mean Pearson r: {np.mean(test_pearson_values):.3f} ± {np.std(test_pearson_values):.3f}")
            print(f"Best channel: {iteration_results[np.argmax(test_pearson_values)]['channel_name']} (r={np.max(test_pearson_values):.3f})")
    
    print(f"\n{'='*80}")
    print("LEAVE-ONE-OUT SUMMARY")
    print(f"{'='*80}")
    
    # Overall summary across all LOO iterations
    all_test_pearson = [r['test_pearson_r'] for r in all_loo_results if not np.isnan(r['test_pearson_r'])]
    if all_test_pearson:
        print(f"Overall LOO Performance (Pearson r): {np.mean(all_test_pearson):.3f} ± {np.std(all_test_pearson):.3f}")
        print(f"Best overall performance: {np.max(all_test_pearson):.3f}")
    
    return {
        'loo_results': all_loo_results,
        'selected_subjects': selected_subjects,
        'n_loo_iterations': len(selected_subjects),
        'overall_performance': {
            'mean_pearson': np.mean(all_test_pearson) if all_test_pearson else 0.0,
            'std_pearson': np.std(all_test_pearson) if all_test_pearson else 0.0,
            'max_pearson': np.max(all_test_pearson) if all_test_pearson else 0.0
        }
    }

def find_subject_files_with_paths(eeg_base, fmri_base, roi_base):
    """Find all subject directories and their corresponding fMRI and ROI files using provided paths."""
    eeg_base = Path(eeg_base)
    fmri_base = Path(fmri_base)
    roi_base = Path(roi_base)
    
    subject_info = []
    
    # Find all subject directories
    subject_dirs = glob.glob(str(eeg_base / "sub-*"))
    
    for subj_dir in subject_dirs:
        subj_dir = Path(subj_dir)
        subj_name = subj_dir.name  # e.g., "sub-056_ses-day2"
        
        # Find eeg_preprocessed.npy file
        eeg_file = subj_dir / "eeg_preprocessed.npy"
        if not eeg_file.exists():
            print(f"Warning: EEG file not found for {subj_name}: {eeg_file}")
            continue
            
        # Find metadata.json file
        metadata_file = subj_dir / "metadata.json"
        if not metadata_file.exists():
            print(f"Warning: Metadata file not found for {subj_name}: {metadata_file}")
            continue
            
        # Extract subject number and session for fMRI file matching
        parts = subj_name.split('_')
        if len(parts) >= 2:
            sub_part = parts[0]  # sub-056
            ses_part = parts[1]  # ses-day2
            
            # Look for corresponding fMRI file
            fmri_pattern = f"{sub_part}_{ses_part}_task-V1LOC_acq-3_dir-AP_bold_fugue_ds_st_mc.nii.gz"
            fmri_file = fmri_base / fmri_pattern
            
            if not fmri_file.exists():
                # Try alternative pattern
                fmri_files = list(fmri_base.glob(f"{sub_part}_{ses_part}_task-V1*_bold*.nii.gz"))
                if fmri_files:
                    fmri_file = fmri_files[0]
                else:
                    print(f"Warning: fMRI file not found for {subj_name}")
                    continue
            
            # Look for corresponding ROI file
            roi_pattern = f"{sub_part}_{ses_part}_roi.nii*"  # Matches .nii or .nii.gz
            roi_files = list(roi_base.glob(roi_pattern))
            
            if not roi_files:
                print(f"Warning: ROI file not found for {subj_name}")
                continue
            roi_file = roi_files[0]  # Take first match
        else:
            print(f"Warning: Cannot parse subject name: {subj_name}")
            continue
            
        subject_info.append({
            'subject_id': subj_name,
            'eeg_file': eeg_file,
            'fmri_file': fmri_file,
            'roi_file': roi_file,
            'metadata_file': metadata_file
        })
    
    return subject_info


def load_subject_data(subject_info, roi_path=None):
    """Load and prepare data for a single subject."""
    subject_id = subject_info['subject_id']
    eeg_file = subject_info['eeg_file']
    fmri_file = subject_info['fmri_file']
    roi_file = subject_info['roi_file']  # Use subject-specific ROI file
    metadata_file = subject_info['metadata_file']
    
    try:
        # Load metadata
        metadata = load_metadata(metadata_file)
        if metadata is None:
            return None, f"Failed to load metadata for {subject_id}"
        
        # Extract channel information
        channel_names = None
        if 'channels' in metadata:
            channel_names = metadata['channels']
        elif 'channel_names' in metadata:
            channel_names = metadata['channel_names']
        elif 'ch_names' in metadata:
            channel_names = metadata['ch_names']
        
        n_bands = metadata.get('n_bands', 10)
        
        # Load EEG data
        eeg_data = np.load(eeg_file)
        
        # Handle different data formats
        if len(eeg_data.shape) == 4:
            n_windows_eeg, n_channels_actual, n_bands_actual, n_timepoints = eeg_data.shape
            eeg_features = eeg_data.reshape(n_windows_eeg, -1)
            n_channels = n_channels_actual
            n_bands = n_bands_actual
        elif len(eeg_data.shape) == 2:
            n_windows_eeg = eeg_data.shape[0]
            actual_features = eeg_data.shape[1]
            if channel_names:
                n_channels = len(channel_names)
            else:
                n_channels = metadata.get('n_channels', 64)
            n_timepoints = actual_features // (n_channels * n_bands)
            eeg_features = eeg_data
        else:
            return None, f"Unsupported EEG data format for {subject_id}: {eeg_data.shape}"
        
        # Validate channel names
        if channel_names and len(channel_names) != n_channels:
            if len(channel_names) > n_channels:
                channel_names = channel_names[:n_channels]
            else:
                channel_names.extend([f"Channel_{i}" for i in range(len(channel_names), n_channels)])
        
        # Load fMRI and ROI data (use subject-specific ROI file)
        roi_img = nib.load(roi_file)
        fmri_img = nib.load(fmri_file)
        fmri_data = fmri_img.get_fdata()
        roi_data = roi_img.get_fdata()
        
        # Create target from ROI
        targets = fmri_data[roi_data == 1].mean(0)
        z_targets = (targets - targets.mean()) / targets.std()
        
        # Match EEG windows length
        if len(z_targets) > n_windows_eeg:
            trimmed_targets = z_targets[-n_windows_eeg:]
        elif len(z_targets) < n_windows_eeg:
            interp_func = interp1d(np.linspace(0, 1, len(z_targets)), z_targets, kind='linear')
            trimmed_targets = interp_func(np.linspace(0, 1, n_windows_eeg))
        else:
            trimmed_targets = z_targets
        
        if np.std(trimmed_targets) == 0:
            return None, f"Zero variance in target for {subject_id}"
        
        # Enhanced metadata
        enhanced_metadata = {
            'subject_id': subject_id,
            'n_channels': n_channels,
            'n_bands': n_bands,
            'n_timepoints': n_timepoints,
            'n_windows': n_windows_eeg,
            'target_std': float(np.std(trimmed_targets)),
            'channel_names': channel_names,
            'original_metadata': metadata
        }
        
        return (eeg_features, trimmed_targets, enhanced_metadata), None
        
    except Exception as e:
        return None, f"Error loading {subject_id}: {str(e)}"

def pearson_scorer(y_true, y_pred):
    """Safe Pearson correlation scorer that handles edge cases."""
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    try:
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    except:
        return 0.0

def load_metadata(metadata_file):
    """Load metadata from JSON file."""
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        print(f"Error loading metadata from {metadata_file}: {e}")
        return None

# ============================================================================
# MAIN FUNCTION: CLUSTERING + TRAINING (MODIFIED FOR ARGPARSE)
# ============================================================================

def clustering_and_training_pipeline(eeg_base, fmri_base, roi_base, results_csv, output_dir, clustering_channel='F8', training_channels=None):
    """Main pipeline: Do clustering first, then train on selected subjects.
    
    Args:
        eeg_base: Path to the EEG data directory
        fmri_base: Path to the fMRI data directory
        roi_base: Path to the ROI definition directory (contains subject-specific ROI files)
        results_csv: CSV output file path for clustering
        output_dir: Directory where output will be saved
        clustering_channel: Channel to use for clustering (e.g., 'F8' or None for all)
        training_channels: List of channels to train on (e.g., ['F8'] or None for all)
    """
    
    # If training_channels not specified, use clustering_channel
    if training_channels is None:
        training_channels = [clustering_channel] if clustering_channel else None
    
    # Configuration - now using function parameters
    OUTPUT_DIR = output_dir
    RESULTS_CSV = results_csv
    
    print("="*80)
    print("CLUSTERING + TRAINING PIPELINE")
    print("="*80)
    
    # Step 1: Clustering Analysis
    print("\n1. CLUSTERING ANALYSIS")
    print("="*40)
    
    if not os.path.exists(RESULTS_CSV):
        print(f"Error: Results file not found: {RESULTS_CSV}")
        return None
    
    # Run clustering on existing results
    clustering_results = enhanced_clustering_with_selection(
        csv_path=RESULTS_CSV,
        channel_name=clustering_channel,  # Use specified channel for clustering
        n_clusters=None,    # auto-determine
        n_to_select=5,      # select 5 representative subjects
        output_dir=os.path.join(OUTPUT_DIR, "clustering_results")
    )
    
    if clustering_results is None:
        print("Clustering failed!")
        return None
    
    selected_subjects = clustering_results['selected_subjects']
    print(f"\nSelected subjects for training: {selected_subjects}")
    
    # Step 2: Load ONLY Selected Subject Data
    print(f"\n2. LOADING SELECTED SUBJECT DATA")
    print("="*40)
    
    # Modified find_subject_files to use the provided paths
    subject_info_list = find_subject_files_with_paths(eeg_base, fmri_base, roi_base)
    cluster_data = {}
    failed_subjects = []
    
    # Filter to only selected subjects
    selected_subject_info = [info for info in subject_info_list if info['subject_id'] in selected_subjects]
    
    print(f"Loading only {len(selected_subject_info)} selected subjects...")
    for subject_info in selected_subject_info:
        print(f"Loading {subject_info['subject_id']}...")
        result, error = load_subject_data(subject_info)
        if result is not None:
            cluster_data[subject_info['subject_id']] = result
        else:
            failed_subjects.append(error)
            print(f"  Failed: {error}")
    
    print(f"Successfully loaded {len(cluster_data)} selected subjects")
    if failed_subjects:
        print(f"Failed to load {len(failed_subjects)} subjects")
    
    # Verify all selected subjects are available
    available_selected = [s for s in selected_subjects if s in cluster_data]
    if len(available_selected) != len(selected_subjects):
        missing = [s for s in selected_subjects if s not in cluster_data]
        print(f"Warning: Missing data for selected subjects: {missing}")
        selected_subjects = available_selected
    
    if len(selected_subjects) < 2:
        print("Error: Need at least 2 subjects for Leave-One-Out!")
        return None
    
    # Step 3: Get Model Parameters
    first_subject_data = next(iter(cluster_data.values()))
    _, _, first_metadata = first_subject_data
    n_channels = first_metadata['n_channels']
    n_bands = first_metadata['n_bands']
    n_timepoints = first_metadata['n_timepoints']
    channel_names = first_metadata['channel_names']
    
    # Convert training_channels to range_channels for training
    range_channels = None
    if training_channels:
        if channel_names:
            range_channels = [channel_names.index(ch) for ch in training_channels if ch in channel_names]
            if not range_channels:
                print(f"Warning: None of the specified training channels ({training_channels}) found in channel names")
                return None
        else:
            print("Warning: No channel names available, cannot train on specific channels")
            return None
    
    # Step 4: Train Models on Selected Cluster
    print(f"\n4. TRAINING ON SELECTED SUBJECTS")
    print("="*40)
    print(f"Training subjects: {selected_subjects}")
    print(f"Training only on channel: {training_channels}")
    
    training_results = leave_one_out_cluster_training(
        cluster_data=cluster_data,
        selected_subjects=selected_subjects,
        n_channels=n_channels,
        n_bands=n_bands,
        n_timepoints=n_timepoints,
        channel_names=channel_names,
        inner_k=5,
        regularization_grid_search=10,
        range_channels=range_channels,  # This will train only on specified channel
        metric='MSE',
        verbose=True,
        estimator=Ridge
    )
    
    # Step 5: Save Results
    print(f"\n5. SAVING RESULTS")
    print("="*40)
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Save LOO results per subject
    if training_results.get('loo_results'):
        loo_df = pd.DataFrame(training_results['loo_results'])
        
        # Group results by test subject
        for test_subject, group in loo_df.groupby('test_subject'):
            # Extract subject parts from subject ID
            parts = test_subject.split('_')
            if len(parts) >= 2:
                sub_part = parts[0]
                ses_part = parts[1]
                # Save per-subject results
                group.to_csv(Path(OUTPUT_DIR) / f"{sub_part}_{ses_part}_results.csv", index=False)
        
        # Save combined results
        channel_suffix = "_".join(training_channels) if training_channels else "all_channels"
        loo_df.to_csv(Path(OUTPUT_DIR) / f'leave_one_out_results_{channel_suffix}.csv', index=False)
        print(f"Saved: leave_one_out_results_{channel_suffix}.csv")
    
    # Save models and metadata
    model_info = {
        'loo_results': training_results,
        'channel_names': channel_names,
        'n_channels': n_channels,
        'n_bands': n_bands,
        'n_timepoints': n_timepoints,
        'selected_subjects': training_results['selected_subjects'],
        'clustering_results': clustering_results,
        'trained_channels': training_channels
    }
    
    # Save per-subject models
    for subject in selected_subjects:
        parts = subject.split('_')
        if len(parts) >= 2:
            sub_part = parts[0]
            ses_part = parts[1]
            subject_output_dir = Path(OUTPUT_DIR) / subject
            subject_output_dir.mkdir(exist_ok=True)
            
            # Save with pickle
            with open(subject_output_dir / f"{sub_part}_{ses_part}_models.pkl", 'wb') as f:
                pickle.dump(model_info, f)
            
            # Save with joblib
            joblib.dump(model_info, subject_output_dir / f"{sub_part}_{ses_part}_models.joblib")
    
    # Save combined model
    channel_suffix = "_".join(training_channels) if training_channels else "all_channels"
    
    # Save with pickle
    with open(Path(OUTPUT_DIR) / f'cluster_models_{channel_suffix}.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    # Save with joblib
    joblib.dump(model_info, Path(OUTPUT_DIR) / f'cluster_models_{channel_suffix}.joblib')
    
    print(f"Saved: cluster_models_{channel_suffix}.pkl and .joblib")

    return {
        'clustering_results': clustering_results,
        'training_results': training_results,
        'output_dir': OUTPUT_DIR
    }
# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Run a clustering and training pipeline using EEG/fMRI data and a specified ROI."
    )
    parser.add_argument("eeg_base", type=str, help="Path to the EEG data directory, from eeg_loader.py.")
    parser.add_argument("fmri_base", type=str, help="Path to the fMRI data directory.")
    parser.add_argument("roi_base", type=str, help="Path to the ROI definition directory (contains subject-specific ROI files).")
    parser.add_argument("results_csv", type=str, help="CSV file path with clustering results.")
    parser.add_argument("output_dir", type=str, help="Directory where output will be saved, including the regressor.")
    parser.add_argument("clustering_channel", type=str, help="EEG channel to use for clustering.", default='Cz')

    args = parser.parse_args()

    print("CLUSTERING + TRAINING PIPELINE")
    print("="*50)
    print(f"Configuration: {args.clustering_channel} clustering → {args.clustering_channel} channel training.")
    print("="*50)

    # Run clustering with specified channel training
    results = clustering_and_training_pipeline(
        eeg_base=args.eeg_base,
        fmri_base=args.fmri_base,
        roi_base=args.roi_base,
        results_csv=args.results_csv,
        output_dir=args.output_dir,
        clustering_channel=args.clustering_channel,
        training_channels=[args.clustering_channel]  # Train on same channel as clustering
    )