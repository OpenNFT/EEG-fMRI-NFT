import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import json
import os
from pathlib import Path

def plot_on_topomap(electrode_scores, subject_data, vlim=None, order=None, axes=None, plot_bar=True):
    if vlim is None:
        vlim = (min(electrode_scores), max(electrode_scores))
    if type(electrode_scores) is pd.core.series.Series and order is None:
        order = list(electrode_scores.index)
    if order is not None:
        order_mapping = {o: e for o, e in zip(order, electrode_scores)}
        electrode_scores = [order_mapping[label] for label in subject_data['mh_ch_names']]
    if axes is None:
        fig, ax = plt.subplots()
    else:
        ax = axes
    im, _ = mne.viz.plot_topomap(
        electrode_scores,
        subject_data['eeg_data'].info,
        axes=ax,
        size=4,
        names=subject_data['mh_ch_names'],
        show=False,
        cmap='viridis',
        vlim=vlim,
    )
    if plot_bar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    if axes is None:
        plt.show()
    return im

def select_grid_results_by_metric(
        grid_search_results, dataset, corr_cutoff=0.55, selection_metric='pearson val', plot=True, axes=None, vlim=None,
        plot_bar=True):
    included_subjects = []
    for subject in grid_search_results['subject_id'].unique():
        max_achieved_score = grid_search_results[grid_search_results['subject_id'] == subject][selection_metric].max()
        if max_achieved_score > corr_cutoff:
            included_subjects.append(subject)
    print(
        f'Subjects:\n{included_subjects}\nlen: {len(included_subjects)},'
        f' {np.round(len(included_subjects) / len(dataset), 3) * 100}% of the total.')
    selected_subjects_results = grid_search_results[grid_search_results['subject_id'].isin(included_subjects)]
    
    numeric_columns = selected_subjects_results.select_dtypes(include=[np.number]).columns.tolist()
    selected_subjects_results_grouped = selected_subjects_results.groupby('channel_name')[numeric_columns].mean()
    
    channel_name = selected_subjects_results_grouped.index[
        np.argmax(selected_subjects_results_grouped['pearson r test'])]
    print(f'Max mean correlation channel: {channel_name}')
    if plot:
        plot_on_topomap(selected_subjects_results_grouped['pearson r test'], dataset[0], axes=axes, vlim=vlim,
                        plot_bar=plot_bar)
    return included_subjects, channel_name

class RealEEGDataLoader:
    """Load actual EEG data from the eeg directory."""
    
    def __init__(self, eeg_dir, results_csv_path):
        self.eeg_dir = Path(eeg_dir)
        self.results_csv_path = results_csv_path
        
        # Load results CSV
        if not Path(results_csv_path).exists():
            raise FileNotFoundError(f"Results CSV not found: {results_csv_path}")
        
        self.results_df = pd.read_csv(results_csv_path)
        print(f"Loaded results for {len(self.results_df['subject_id'].unique())} subjects")
        
        # Try to find metadata files to get channel information
        self.metadata_files = self.find_metadata_files()
        self.dataset = self.load_dataset_structure()
    
    def find_metadata_files(self):
        """Find metadata files in the EEG directory."""
        metadata_files = {}
        
        if not self.eeg_dir.exists():
            print(f"EEG directory does not exist: {self.eeg_dir}")
            return metadata_files
        
        # Look for metadata.json files
        json_files = list(self.eeg_dir.glob("**/metadata.json"))
        joblib_files = list(self.eeg_dir.glob("**/*_metadata.joblib"))
        
        print(f"Found {len(json_files)} metadata.json files")
        print(f"Found {len(joblib_files)} metadata.joblib files")
        
        # Process JSON files
        for file in json_files:
            subject_id = self.extract_subject_id_from_path(file.parent)
            if subject_id:
                metadata_files[subject_id] = file
        
        # Process joblib files if no JSON found
        if not metadata_files:
            for file in joblib_files:
                subject_id = self.extract_subject_id_from_path(file)
                if subject_id:
                    metadata_files[subject_id] = file
        
        print(f"Mapped metadata files to {len(metadata_files)} subjects")
        return metadata_files
    
    def extract_subject_id_from_path(self, file_path):
        """Extract subject ID from file path."""
        path_str = str(file_path)
        
        import re
        # Try to match sub-XXX_ses-dayX pattern first
        match = re.search(r'sub-\d+_ses-day\d+', path_str)
        if match:
            return match.group()
        
        # Try to match just sub-XXX pattern
        match = re.search(r'sub-\d+', path_str)
        if match:
            return match.group()
        
        return None
    
    def load_metadata_file(self, file_path):
        """Load metadata from either JSON or joblib file."""
        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            elif file_path.suffix == '.joblib':
                import joblib
                return joblib.load(file_path)
            else:
                return None
        except Exception as e:
            print(f"Error loading metadata from {file_path}: {e}")
            return None
    
    def create_dataset_from_metadata(self, metadata, subject_id):
        """Create dataset structure from metadata."""
        try:
            # Extract channel information from metadata
            channel_names = None
            if 'channels' in metadata:
                channel_names = metadata['channels']
            elif 'channel_names' in metadata:
                channel_names = metadata['channel_names']
            elif 'ch_names' in metadata:
                channel_names = metadata['ch_names']
            
            if not channel_names:
                # Fallback: get unique channel names from results CSV
                subject_results = self.results_df[self.results_df['subject_id'] == subject_id]
                if not subject_results.empty:
                    channel_names = sorted(subject_results['channel_name'].unique())
                else:
                    # Use all unique channels from CSV
                    channel_names = sorted(self.results_df['channel_name'].unique())
            
            if not channel_names:
                print(f"No channel names found for subject {subject_id}")
                return None
            
            print(f"Found {len(channel_names)} channels for {subject_id}")
            
            # Create MNE info structure
            info = mne.create_info(channel_names, 1000, ch_types='eeg')
            
            # Try to set standard montage
            try:
                montage = mne.channels.make_standard_montage('standard_1020')
                
                # Find common channels between our data and standard montage
                common_channels = [ch for ch in channel_names if ch in montage.ch_names]
                
                if len(common_channels) >= 8:  # Need at least 8 channels for topomap
                    print(f"Using {len(common_channels)} standard channels for montage")
                    # Filter to only use standard channels
                    info = mne.create_info(common_channels, 1000, ch_types='eeg')
                    info.set_montage(montage, on_missing='ignore')
                    channel_names = common_channels
                else:
                    print(f"Only {len(common_channels)} standard channels found, using all channels")
                    # Try to set montage with available channels
                    info.set_montage(montage, on_missing='ignore')
                    
            except Exception as e:
                print(f"Could not set montage: {e}")
                # Continue without montage
            
            # Create dummy evoked data for topomap plotting
            n_channels = len(channel_names)
            dummy_data = np.zeros((n_channels, 1))
            evoked = mne.EvokedArray(dummy_data, info)
            
            subject_data = {
                'eeg_data': evoked,
                'mh_ch_names': channel_names,
                'f_features': 1,
                'subject_id': subject_id
            }
            
            return subject_data
            
        except Exception as e:
            print(f"Error creating dataset from metadata for {subject_id}: {e}")
            return None
    
    def load_dataset_structure(self):
        """Load dataset structure from metadata files."""
        if not self.metadata_files:
            print("No metadata files found, trying to create from CSV results...")
            return self.create_dataset_from_csv()
        
        # Try to load from first available metadata file
        for subject_id, file_path in self.metadata_files.items():
            print(f"Loading dataset structure from {subject_id}")
            metadata = self.load_metadata_file(file_path)
            if metadata:
                subject_data = self.create_dataset_from_metadata(metadata, subject_id)
                if subject_data:
                    print(f"Successfully created dataset structure: {len(subject_data['mh_ch_names'])} channels")
                    return [subject_data]
        
        # Fallback to CSV-based creation
        print("Failed to load from metadata, creating from CSV...")
        return self.create_dataset_from_csv()
    
    def create_dataset_from_csv(self):
        """Create dataset structure from CSV results only."""
        try:
            # Get all unique channel names from results
            channel_names = sorted(self.results_df['channel_name'].unique())
            
            if not channel_names:
                print("No channel names found in CSV")
                return None
            
            print(f"Creating dataset from CSV with {len(channel_names)} channels")
            
            # Create MNE info structure
            info = mne.create_info(channel_names, 1000, ch_types='eeg')
            
            # Try to set standard montage
            try:
                montage = mne.channels.make_standard_montage('standard_1020')
                common_channels = [ch for ch in channel_names if ch in montage.ch_names]
                
                if len(common_channels) >= 8:
                    print(f"Using {len(common_channels)} standard channels")
                    info = mne.create_info(common_channels, 1000, ch_types='eeg')
                    info.set_montage(montage, on_missing='ignore')
                    channel_names = common_channels
                else:
                    print(f"Only {len(common_channels)} standard channels, using all")
                    info.set_montage(montage, on_missing='ignore')
                    
            except Exception as e:
                print(f"Could not set montage: {e}")
            
            # Create dummy evoked data
            n_channels = len(channel_names)
            dummy_data = np.zeros((n_channels, 1))
            evoked = mne.EvokedArray(dummy_data, info)
            
            subject_data = {
                'eeg_data': evoked,
                'mh_ch_names': channel_names,
                'f_features': 1,
                'subject_id': 'combined'
            }
            
            return [subject_data]
            
        except Exception as e:
            print(f"Error creating dataset from CSV: {e}")
            return None
    
    def create_two_panel_plot(self, 
                             vlim1=None, 
                             vlim2=None,
                             cutoff=0.3,
                             selection_metric='pearson r test',
                             test_metric='pearson r test',
                             save_path=None,
                             figsize=(12, 5)):
        """Create a two-panel plot: A) All subjects, B) Subjects above cutoff."""
        
        if self.dataset is None:
            print("Cannot create plot: No EEG dataset available")
            return None
        
        # Get numeric columns
        numeric_columns = self.results_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Panel A: All subjects average
        grouped_corr = self.results_df.groupby('channel_name')[numeric_columns].mean()
        
        # Set vlim if not provided
        if vlim1 is None:
            vlim1 = (grouped_corr[test_metric].min(), grouped_corr[test_metric].max())
        if vlim2 is None:
            vlim2 = vlim1
        
        # Print some statistics
        print(f"All subjects average {test_metric}: {grouped_corr[test_metric].mean():.4f}")
        print(f"Range: {grouped_corr[test_metric].min():.4f} to {grouped_corr[test_metric].max():.4f}")
        
        # Print F8 average correlation for all subjects if available
        if 'F8' in grouped_corr.index:
            f8_mean = grouped_corr.loc['F8', test_metric]
            print(f"F8 average correlation (all subjects): {f8_mean:.4f}")
        
        # Create the figure
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        print("Creating two-panel topomap...")
        
        # Panel A: All subjects - no colorbar
        try:
            plot_on_topomap(grouped_corr[test_metric], self.dataset[0], axes=axes[0], vlim=vlim1, plot_bar=False)
            axes[0].set_title('A. All Subjects Average', fontsize=14, fontweight='bold')
        except Exception as e:
            print(f"Error in Panel A: {e}")
            axes[0].text(0.5, 0.5, f'Panel A\nError: {str(e)[:50]}...', 
                        transform=axes[0].transAxes, ha='center', va='center', fontsize=10)
        
        # Panel B: Selected subjects above cutoff - with colorbar
        try:
            included_subjects, best_channel = select_grid_results_by_metric(
                self.results_df,
                self.dataset, 
                corr_cutoff=cutoff, 
                selection_metric=selection_metric, 
                axes=axes[1],
                vlim=vlim2, 
                plot_bar=True
            )
            axes[1].set_title(f'B. Subjects with {selection_metric} > {cutoff}', fontsize=14, fontweight='bold')
            print(f"Best channel overall: {best_channel}")
            
        except Exception as e:
            print(f"Error in Panel B: {e}")
            axes[1].text(0.5, 0.5, f'Panel B\nError: {str(e)[:50]}...', 
                        transform=axes[1].transAxes, ha='center', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            # Ensure output directory exists
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        return fig

def main(eeg_samples_dir, results_csv_path, output_path, cutoff):
    """Main function to load real EEG data and create plots."""
    
    # Create output directory
    output_path = Path(output_path)
    if output_path.suffix:
        # If output_path has an extension, it's a file path
        output_dir = output_path.parent
        save_path = output_path
    else:
        # If no extension, treat as directory
        output_dir = output_path
        save_path = output_path / "topomaps_comparison.png"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        loader = RealEEGDataLoader(eeg_samples_dir, results_csv_path)
        
        if loader.dataset is None:
            print("Failed to load EEG data structure")
            return
        
        print(f"Creating topomaps with cutoff: {cutoff}")
        fig = loader.create_two_panel_plot(
            cutoff=cutoff,
            save_path=save_path
        )
        
        if fig is not None:
            print("Successfully created topomaps")
        else:
            print("Failed to create topomaps")
            
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw average individual regressor topomaps for all EEG runs."
    )
    parser.add_argument("eeg_samples_dir", type=str, help="Path to the directory containing samples prepared by eeg_loader.py.")
    parser.add_argument("results_csv_path", type=str, help="Path to the CSV file generated by train_for_all.py.")
    parser.add_argument("output_path", type=str, help="Output path for the topomap file (can be file or directory).")
    parser.add_argument("cutoff", type=float, help="Minimum correlation threshold for filtered topomap.")
    
    args = parser.parse_args()

    main(args.eeg_samples_dir, args.results_csv_path, args.output_path, args.cutoff)