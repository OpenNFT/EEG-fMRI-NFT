import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS, TSNE
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm.autonotebook import tqdm
import argparse
import os
from pathlib import Path

def load_csv_data_for_clustering(csv_path, channel_name=None):
    """
    Load your CSV data and prepare it for clustering.
    """
    
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    print(f"Unique subjects: {len(df['subject_id'].unique())}")
    print(f"Unique channels: {len(df['channel_name'].unique())}")
    
    # Filter for specific channel if requested
    if channel_name is not None:
        df = df[df['channel_name'] == channel_name].copy()
        
        if len(df) == 0:
            print(f"ERROR: No data found for channel {channel_name}")
            available_channels = sorted(pd.read_csv(csv_path)['channel_name'].unique())
            print(f"Available channels: {available_channels[:10]}...")
            return None, None, None
    
    # Performance metrics for clustering
    performance_columns = [
        'pearson r test', 'pearson p test', 'MSE test', 'nMSE test', 'r2 test',
        'MSE val', 'r2 val', 'pearson val'
    ]
    
    available_columns = [col for col in performance_columns if col in df.columns]
    print(f"Available performance columns: {available_columns}")
    
    if channel_name is None:
        # Multi-channel: create pivot table (subjects x channels)
        print("Creating multi-channel analysis (subjects x channels)")
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
        print(f"Creating single-channel analysis for {channel_name}")
        df_sorted = df.sort_values('subject_id')
        subjects = df_sorted['subject_id'].tolist()
        data_matrix = df_sorted[available_columns].values
        feature_names = available_columns
        
        # Handle missing values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        data_matrix = imputer.fit_transform(data_matrix)
    
    print(f"Final data matrix: {len(subjects)} subjects × {len(feature_names)} features")
    print(f"Data shape: {data_matrix.shape}")
    
    return data_matrix, subjects, feature_names

def create_distance_matrix(data_matrix, metric='cosine'):
    """Create subject×subject distance matrix."""
    
    distances = pdist(data_matrix, metric=metric)
    distance_matrix = squareform(distances)
    
    return distance_matrix

def plot_dendrogram(model, subject_labels=None, save_path=None, **kwargs):
    """
    Plot dendrogram from clustering model.
    """
    
    # Create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, labels=subject_labels, **kwargs)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Subject ID')
    plt.ylabel('Distance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dendrogram saved to: {save_path}")
    
    plt.show()

def calculate_wcss(distance_matrix, labels, n_clusters):
    """
    Compute Within-Cluster Sum of Squares from distance matrix.
    """
    wcss = 0
    
    for i in range(n_clusters):
        cluster_points = np.where(labels == i)[0]
        if len(cluster_points) > 1:
            # Get pairwise distances within cluster
            intra_cluster_distances = distance_matrix[np.ix_(cluster_points, cluster_points)]
            wcss += np.sum(intra_cluster_distances) / (2 * len(cluster_points))
    
    return wcss

def projection_plot_clusters(distance_matrix, clusters, subjects, channel_name=None, 
                           special_markers=(), save_path=None):
    """Create 2D projection plot of clusters using MDS or t-SNE."""
    
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
                       color='red', s=200, edgecolors='black', label='Special subjects')
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
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Projection plot saved to: {save_path}")
    
    plt.show()

def agglomerative_clustering_csv(n_clusters, data_matrix, subjects, 
                                subject_labels=None, linkage='average', 
                                metric='cosine', plot=True, channel_name=None,
                                special_markers=(), save_dir=None):
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
    print(f"WCSS (Within-Cluster Sum of Squares): {wcss:.3f}")
    
    if n_clusters > 1:
        sil_score = silhouette_score(distance_matrix, clusters, metric='precomputed')
        print(f"Silhouette Score: {sil_score:.3f}")
    
    # Print cluster assignments
    print(f"\nCluster Assignments:")
    unique_clusters = np.unique(clusters)
    for cluster_id in unique_clusters:
        subjects_in_cluster = [subjects[i] for i in range(len(subjects)) 
                              if clusters[i] == cluster_id]
        print(f"Cluster {cluster_id}: {subjects_in_cluster}")
    
    # Create plots
    if plot:
        channel_prefix = channel_name if channel_name else 'all_channels'
        
        dendrogram_path = None
        projection_path = None
        
        if save_dir:
            dendrogram_path = Path(save_dir) / f"{channel_prefix}_dendrogram.png"
            projection_path = Path(save_dir) / f"{channel_prefix}_projection.png"
        
        plot_dendrogram(clustering, subject_labels=subject_labels, save_path=dendrogram_path)
        projection_plot_clusters(distance_matrix, clusters, subjects, 
                                channel_name=channel_name, special_markers=special_markers,
                                save_path=projection_path)
    
    return wcss, clustering, clusters, distance_matrix

def find_optimal_clusters(data_matrix, subjects, max_clusters=10, 
                         linkage='average', metric='cosine', channel_name=None,
                         save_dir=None):
    """Find optimal number of clusters using silhouette score."""
    
    max_clusters = min(max_clusters, len(subjects) - 1)
    wcss_scores = []
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)
    
    print(f"Testing cluster range: {list(cluster_range)}")
    
    # Create distance matrix once
    distance_matrix = create_distance_matrix(data_matrix, metric=metric)
    
    for n_clusters in tqdm(cluster_range, desc="Testing cluster numbers"):
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
        
        print(f"n_clusters={n_clusters}: WCSS={wcss:.3f}, Silhouette={sil_score:.3f}")
    
    # Plot optimization curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # WCSS (Elbow method)
    axes[0].plot(cluster_range, wcss_scores, 'bo-')
    axes[0].set_title('Elbow Method (WCSS)')
    axes[0].set_xlabel('Number of Clusters')
    axes[0].set_ylabel('WCSS')
    axes[0].grid(True)
    
    # Silhouette Score (higher is better)
    axes[1].plot(cluster_range, silhouette_scores, 'ro-')
    axes[1].set_title('Silhouette Score')
    axes[1].set_xlabel('Number of Clusters')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].grid(True)
    
    title = 'Cluster Optimization'
    if channel_name:
        title += f' ({channel_name})'
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_dir:
        channel_prefix = channel_name if channel_name else 'all_channels'
        optimization_path = Path(save_dir) / f"{channel_prefix}_optimization.png"
        plt.savefig(optimization_path, dpi=300, bbox_inches='tight')
        print(f"Optimization plot saved to: {optimization_path}")
    
    plt.show()
    
    # Find optimal number of clusters
    optimal_silhouette = cluster_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal clusters (silhouette): {optimal_silhouette}")
    
    return {
        'cluster_range': list(cluster_range),
        'wcss_scores': wcss_scores,
        'silhouette_scores': silhouette_scores,
        'optimal_silhouette': optimal_silhouette,
    }

def main_clustering_analysis(csv_path, output_dir, channel_name=None, max_clusters=10, 
                           special_subjects=None):
    """Main function to run clustering analysis."""
    
    print(f"Starting clustering analysis...")
    print(f"CSV path: {csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Channel: {channel_name if channel_name else 'All channels'}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_matrix, subjects, feature_names = load_csv_data_for_clustering(
        csv_path, channel_name=channel_name
    )
    
    if data_matrix is None:
        print("Failed to load data. Exiting.")
        return None
    
    # Parse special subjects
    special_markers = []
    if special_subjects:
        special_markers = [s.strip() for s in special_subjects.split(',')]
        print(f"Special subjects to highlight: {special_markers}")
    
    # Find optimal number of clusters
    print(f"\nFinding optimal number of clusters (max: {max_clusters})...")
    optimization_results = find_optimal_clusters(
        data_matrix, subjects, max_clusters=max_clusters,
        channel_name=channel_name, save_dir=output_dir
    )
    
    # Perform clustering with optimal number
    optimal_n_clusters = optimization_results['optimal_silhouette']
    print(f"\nPerforming clustering with {optimal_n_clusters} clusters...")
    
    wcss, clustering, clusters, distance_matrix = agglomerative_clustering_csv(
        n_clusters=optimal_n_clusters,
        data_matrix=data_matrix,
        subjects=subjects,
        linkage='average',
        metric='cosine',
        plot=True,
        channel_name=channel_name,
        special_markers=special_markers,
        save_dir=output_dir
    )
    
    # Save results
    channel_prefix = channel_name if channel_name else 'all_channels'
    
    # Save cluster assignments
    results_df = pd.DataFrame({
        'subject_id': subjects,
        'cluster': clusters
    })
    cluster_file = output_dir / f"{channel_prefix}_clusters.csv"
    results_df.to_csv(cluster_file, index=False)
    print(f"Cluster assignments saved to: {cluster_file}")
    
    # Save distance matrix
    distance_df = pd.DataFrame(distance_matrix, index=subjects, columns=subjects)
    distance_file = output_dir / f"{channel_prefix}_distance_matrix.csv"
    distance_df.to_csv(distance_file)
    print(f"Distance matrix saved to: {distance_file}")
    
    # Save optimization results
    optimization_df = pd.DataFrame({
        'n_clusters': optimization_results['cluster_range'],
        'wcss': optimization_results['wcss_scores'],
        'silhouette_score': optimization_results['silhouette_scores']
    })
    optimization_file = output_dir / f"{channel_prefix}_optimization_results.csv"
    optimization_df.to_csv(optimization_file, index=False)
    print(f"Optimization results saved to: {optimization_file}")
    
    print(f"\nAll results saved to: {output_dir}")
    
    return {
        'clusters': clusters,
        'subjects': subjects,
        'distance_matrix': distance_matrix,
        'clustering': clustering,
        'optimization': optimization_results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform clustering analysis on EEG-fMRI regression results."
    )
    parser.add_argument("csv_path", type=str, 
                       help="Path to the CSV file with regression results.")
    parser.add_argument("output_dir", type=str,
                       help="Directory to save clustering results.")
    parser.add_argument("--channel", type=str, default=None,
                       help="Specific channel name to analyze (default: analyze all channels).")
    parser.add_argument("--max_clusters", type=int, default=10,
                       help="Maximum number of clusters to test (default: 10).")
    parser.add_argument("--special_subjects", type=str, default=None,
                       help="Comma-separated list of special subjects to highlight in plots.")
    
    args = parser.parse_args()
    
    # Run the analysis
    results = main_clustering_analysis(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        channel_name=args.channel,
        max_clusters=args.max_clusters,
        special_subjects=args.special_subjects
    )