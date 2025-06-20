import numpy as np
import nibabel as nib
import argparse
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import tqdm
from scipy.interpolate import interp1d
import os
import glob
import pandas as pd
import pickle
import json
from pathlib import Path

# ========== Utility Scorers ==========
def pearson_scorer(y_true, y_pred):
    """Safe Pearson correlation scorer that handles edge cases."""
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0  # Return 0 if no variance
    try:
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    except:
        return 0.0

# ========== Regularization Grid ==========
def get_regularization_grid(X, num=10):
    alphas = np.logspace(-6, 6, num=num)
    return {'alpha': alphas}

# ========== Load Metadata ==========
def load_metadata(metadata_file):
    """Load metadata from JSON file."""
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        print(f"Error loading metadata from {metadata_file}: {e}")
        return None

# ========== Reshape per channel ==========
def reshape_data_for_inference(data, channel, n_bands, n_timepoints):
    """
    Reshape data for a specific channel.
    
    Args:
        data: shape (n_windows, n_features) where n_features = n_channels * n_bands * n_timepoints
        channel: channel index to extract
        n_bands: number of frequency bands
        n_timepoints: number of timepoints per band
        
    Returns:
        reshaped_data: shape (n_windows, n_bands * n_timepoints)
    """
    n_windows = data.shape[0]
    features_per_channel = n_bands * n_timepoints
    start_idx = channel * features_per_channel
    end_idx = (channel + 1) * features_per_channel
    return data[:, start_idx:end_idx]

def grid_search_with_alpha_grid(channel, estimator, inner_splitter, metric, regularization_grid_search, train_data,
                                train_targets, n_bands, n_timepoints, verbose=0):
    """Grid search for a single channel's data."""
    one_ch_data = reshape_data_for_inference(train_data, channel, n_bands, n_timepoints)
    
    if isinstance(regularization_grid_search, int):
        alpha_grid = get_regularization_grid(one_ch_data, regularization_grid_search)
    else:
        alpha_grid = regularization_grid_search
        
    grid_search = GridSearchCV(
        estimator=estimator(),
        param_grid=alpha_grid,
        cv=inner_splitter,
        verbose=verbose,
        scoring={
            'pearson': make_scorer(pearson_scorer),
            'MSE': make_scorer(mean_squared_error, greater_is_better=False),
            'r2': make_scorer(r2_score)
        },
        refit=metric,
        n_jobs=-1,
    )
    grid_search.fit(one_ch_data, train_targets)
    return grid_search

def train_efp_model(
        data,  # shape: (n_windows, n_features) where n_features = n_channels * n_bands * n_timepoints
        target,  # shape: (n_windows,)
        n_channels,  # number of channels
        n_bands,  # number of frequency bands
        n_timepoints,  # number of timepoints per band
        channel_names=None,  # list of channel names
        inner_k=5,
        outer_test_ratio=0.2,
        regularization_grid_search=10,
        range_channels=None,
        metric='MSE',
        verbose=False,
        estimator=None,
):
    """
    Train EFP model for each channel.
    
    Args:
        data: shape (n_windows, n_features)
        target: shape (n_windows,)
        n_channels: number of channels
        n_bands: number of frequency bands
        n_timepoints: number of timepoints per band
        channel_names: list of channel names (optional)
    """
    # Validate input data
    if np.std(target) == 0:
        raise ValueError("Target variable has zero variance. Cannot train model with constant target values.")
    
    n_windows = data.shape[0]
    inner_splitter = KFold(n_splits=inner_k, shuffle=False)
    test_set_index = int(n_windows * (1 - outer_test_ratio))

    train_data = data[:test_set_index]
    test_data = data[test_set_index:]
    train_targets = target[:test_set_index]
    test_targets = target[test_set_index:]

    # Additional validation
    if np.std(train_targets) == 0 or np.std(test_targets) == 0:
        raise ValueError("Training or test targets have zero variance.")

    results = []
    models = []

    if estimator is None:
        estimator = Ridge

    if range_channels is None:
        range_channels = range(n_channels)

    for channel in tqdm.tqdm(range_channels, desc='Training a regressor for each channel', disable=not verbose):
        # Get channel name if available
        channel_name = channel_names[channel] if channel_names and channel < len(channel_names) else f"Channel_{channel}"
        
        # Grid search for this channel (reshaping happens inside)
        grid_search = grid_search_with_alpha_grid(
            channel,
            estimator, 
            inner_splitter, 
            metric,
            regularization_grid_search,
            train_data,
            train_targets,
            n_bands,
            n_timepoints
        )
        
        # Reshape test data for this channel
        test_ch_data = reshape_data_for_inference(test_data, channel, n_bands, n_timepoints)

        # Predict on test set
        test_results = grid_search.predict(test_ch_data)

        # Compute metrics with error handling
        try:
            pearson_r, pearson_p = pearsonr(test_targets, test_results)
            if np.isnan(pearson_r):
                pearson_r, pearson_p = 0.0, 1.0
        except:
            pearson_r, pearson_p = 0.0, 1.0
            
        MSE = mean_squared_error(test_targets, test_results)
        r2 = r2_score(test_targets, test_results)
        
        # Safe nMSE calculation
        target_std = np.std(test_targets)
        nMSE = MSE / target_std if target_std > 0 else np.inf

        results.append({
            'channel': channel,
            'channel_name': channel_name,
            'pearson r test': pearson_r,
            'pearson p test': pearson_p,
            'MSE test': MSE,
            'nMSE test': nMSE,
            'r2 test': r2,
            'MSE val': -np.mean(grid_search.cv_results_['mean_test_MSE']) if 'mean_test_MSE' in grid_search.cv_results_ else np.nan,
            'r2 val': np.mean(grid_search.cv_results_['mean_test_r2']) if 'mean_test_r2' in grid_search.cv_results_ else np.nan,
            'pearson val': np.mean(grid_search.cv_results_['mean_test_pearson']) if 'mean_test_pearson' in grid_search.cv_results_ else np.nan,
            'best_alpha': grid_search.best_params_['alpha'] if hasattr(grid_search, 'best_params_') else np.nan,
        })

        models.append(grid_search.best_estimator_)

    return results, models

# Fixed find_subject_files function
def find_subject_files(base_dir_eeg, base_dir_mri, roi_dir):
    """Find all subject directories and their corresponding fMRI and ROI files."""
    # Convert to Path objects if they aren't already
    eeg_base = Path(base_dir_eeg)
    fmri_base = Path(base_dir_mri)
    roi_base = Path(roi_dir)
    
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
            'roi_file': roi_file,  # Now per-subject ROI file
            'metadata_file': metadata_file
        })
    
    return subject_info

def process_single_subject(subject_info, output_dir, verbose=True):
    """Process a single subject and save results."""
    subject_id = subject_info['subject_id']
    eeg_file = subject_info['eeg_file']
    fmri_file = subject_info['fmri_file']
    roi_file = subject_info['roi_file']  # Now from subject_info
    metadata_file = subject_info['metadata_file']
    
    print(f"\n{'='*80}")
    print(f"Processing Subject: {subject_id}")
    print(f"{'='*80}")
    print(f"EEG file: {eeg_file}")
    print(f"fMRI file: {fmri_file}")
    print(f"ROI file: {roi_file}")  # Print ROI file
    print(f"Metadata file: {metadata_file}")
    
    try:
        # Load metadata
        print("\n1. Loading metadata...")
        metadata = load_metadata(metadata_file)
        if metadata is None:
            return None, f"Failed to load metadata for {subject_id}"
        
        # Extract channel information from metadata
        channel_names = None
        if 'channels' in metadata:
            channel_names = metadata['channels']
        elif 'channel_names' in metadata:
            channel_names = metadata['channel_names']
        elif 'ch_names' in metadata:
            channel_names = metadata['ch_names']
        else:
            print("Warning: No channel names found in metadata, will use default naming")
        
        # Extract other parameters from metadata if available
        n_bands = metadata.get('n_bands', 10)  
        print(f"Found {len(channel_names) if channel_names else 'unknown'} channels")
        print(f"Number of frequency bands: {n_bands}")
        if channel_names:
            print(f"Channel names: {channel_names[:5]}{'...' if len(channel_names) > 5 else ''}")
        
        # Load EEG data
        print("\n2. Loading EEG data...")
        eeg_data = np.load(eeg_file)
        print(f"EEG data shape: {eeg_data.shape}")
        
        # Handle different data formats
        if len(eeg_data.shape) == 4:
            # Format: (n_windows, n_channels, n_bands, n_timepoints)
            n_windows_eeg, n_channels_actual, n_bands_actual, n_timepoints = eeg_data.shape
            print(f"4D format detected: (windows={n_windows_eeg}, channels={n_channels_actual}, bands={n_bands_actual}, timepoints={n_timepoints})")
            
            # Reshape to 2D format expected by training function
            eeg_features = eeg_data.reshape(n_windows_eeg, -1)
            print(f"Reshaped to: {eeg_features.shape}")
            
            # Update parameters based on actual data
            n_channels = n_channels_actual
            n_bands = n_bands_actual
            
        elif len(eeg_data.shape) == 2:
            # Format: (n_windows, n_features) - already flattened
            n_windows_eeg = eeg_data.shape[0]
            actual_features = eeg_data.shape[1]
            
            # Determine n_channels from channel_names if available
            if channel_names:
                n_channels = len(channel_names)
            else:
                # Try to infer from metadata or use default
                n_channels = metadata.get('n_channels', 64)  # Default to 64 if not specified
            
            n_timepoints = actual_features // (n_channels * n_bands)
            eeg_features = eeg_data
            print(f"2D format detected: (windows={n_windows_eeg}, features={actual_features})")
            
        else:
            return None, f"Unsupported EEG data format for {subject_id}: {eeg_data.shape}"
        
        print(f"Final dimensions:")
        print(f"  - Windows: {n_windows_eeg}")
        print(f"  - Channels: {n_channels}")
        print(f"  - Bands: {n_bands}")
        print(f"  - Timepoints: {n_timepoints}")
        print(f"  - Total features: {eeg_features.shape[1]}")
        
        # Validate dimensions
        expected_features = n_channels * n_bands * n_timepoints
        if eeg_features.shape[1] != expected_features:
            print(f"Warning: Feature dimension mismatch for {subject_id}: expected {expected_features}, got {eeg_features.shape[1]}")
            # Try to adjust n_timepoints to match actual data
            n_timepoints = eeg_features.shape[1] // (n_channels * n_bands)
            expected_features = n_channels * n_bands * n_timepoints
            print(f"Adjusted timepoints to {n_timepoints}, new expected features: {expected_features}")
            
            if eeg_features.shape[1] != expected_features:
                return None, f"Cannot resolve feature dimension mismatch for {subject_id}"
        
        # Validate channel names
        if channel_names and len(channel_names) != n_channels:
            print(f"Warning: Channel names count ({len(channel_names)}) doesn't match n_channels ({n_channels})")
            if len(channel_names) > n_channels:
                channel_names = channel_names[:n_channels]
            else:
                # Pad with default names
                channel_names.extend([f"Channel_{i}" for i in range(len(channel_names), n_channels)])
        
        # Load fMRI and ROI data
        print("\n3. Loading fMRI and ROI data...")
        roi_img = nib.load(roi_file)  # Changed from roi_path to roi_file
        fmri_img = nib.load(fmri_file)
    
        
        fmri_data = fmri_img.get_fdata()
        roi_data = roi_img.get_fdata()
        
        print(f"ROI data shape: {roi_data.shape}")
        print(f"fMRI data shape: {fmri_data.shape}")
        
        # Create target from ROI
        print("\n4. Creating target variable...")
        targets = fmri_data[roi_data == 1].mean(0)
        z_targets = (targets - targets.mean()) / targets.std()
        
        # Match EEG windows length
        if len(z_targets) > n_windows_eeg:
            trimmed_targets = z_targets[-n_windows_eeg:]
        elif len(z_targets) < n_windows_eeg:
            # Interpolate if fMRI has fewer timepoints
            interp_func = interp1d(np.linspace(0, 1, len(z_targets)), z_targets, kind='linear')
            trimmed_targets = interp_func(np.linspace(0, 1, n_windows_eeg))
        else:
            trimmed_targets = z_targets
            
        print(f"Target shape: {trimmed_targets.shape}")
        print(f"Target std: {np.std(trimmed_targets):.6f}")
        
        if np.std(trimmed_targets) == 0:
            return None, f"Zero variance in target for {subject_id}"
        
        # Train model
        print("\n5. Training EFP model...")
        results, models = train_efp_model(
            data=eeg_features,
            target=trimmed_targets,
            n_channels=n_channels,
            n_bands=n_bands,
            n_timepoints=n_timepoints,
            channel_names=channel_names,
            inner_k=5,
            outer_test_ratio=0.2,
            regularization_grid_search=10,
            metric='MSE',
            verbose=verbose,
            estimator=Ridge
        )
        
        # Save results
        print("\n6. Saving results...")
        subject_output_dir = Path(output_dir) / subject_id
        subject_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results as DataFrame
        results_df = pd.DataFrame(results)
        results_df.to_csv(subject_output_dir / 'results.csv', index=False)
        
        # Save models with channel names
        model_info = {
            'models': models,
            'channel_names': channel_names,
            'n_channels': n_channels,
            'n_bands': n_bands,
            'n_timepoints': n_timepoints
        }
        with open(subject_output_dir / 'models.pkl', 'wb') as f:
            pickle.dump(model_info, f)
            
        # Save enhanced metadata
        enhanced_metadata = {
            'subject_id': subject_id,
            'eeg_file': str(eeg_file),
            'fmri_file': str(fmri_file),
            'metadata_file': str(metadata_file),
            'n_channels': n_channels,
            'n_bands': n_bands,
            'n_timepoints': n_timepoints,
            'n_windows': n_windows_eeg,
            'target_std': float(np.std(trimmed_targets)),
            'channel_names': channel_names,
            'original_metadata': metadata
        }
        
        with open(subject_output_dir / 'enhanced_metadata.pkl', 'wb') as f:
            pickle.dump(enhanced_metadata, f)
        
        # Print summary
        print(f"\n7. Results Summary for {subject_id}:")
        print("-" * 60)
        metrics = ['pearson r test', 'MSE test', 'r2 test']
        for metric in metrics:
            values = [r[metric] for r in results if not np.isnan(r[metric]) and not np.isinf(r[metric])]
            if values:
                print(f"{metric:<20}: mean={np.mean(values):>8.3f}, std={np.std(values):>8.3f}")
        
        # Show best performing channels
        print(f"\nTop 5 channels by Pearson correlation:")
        sorted_results = sorted(results, key=lambda x: x['pearson r test'], reverse=True)
        for i, result in enumerate(sorted_results[:5]):
            print(f"  {i+1}. {result['channel_name']}: r={result['pearson r test']:.3f}")
        
        return results, None
        
    except Exception as e:
        error_msg = f"Error processing {subject_id}: {str(e)}"
        print(f"\n{error_msg}")
        import traceback
        traceback.print_exc()
        return None, error_msg

# Fixed main function
def main(base_dir_eeg, base_dir_mri, roi_dir, out_dir):
    # Convert arguments to Path objects
    base_dir_eeg = Path(base_dir_eeg)
    base_dir_mri = Path(base_dir_mri)
    roi_dir = Path(roi_dir)
    OUTPUT_DIR = Path(out_dir)
    
    print("="*80)
    print("Multi-Subject EEG-fMRI Regressor Training Pipeline")
    print("="*80)
    
    # Find all subjects - Now passing all 3 arguments correctly
    print("\n1. Finding subject files...")
    subject_info_list = find_subject_files(base_dir_eeg, base_dir_mri, roi_dir)
    print(f"Found {len(subject_info_list)} subjects to process")
    
    for info in subject_info_list:
        print(f"  - {info['subject_id']}")
    
    if not subject_info_list:
        print("No subjects found! Check your directory structure.")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each subject
    print(f"\n2. Processing subjects...")
    all_results = []
    errors = []
    
    for i, subject_info in enumerate(subject_info_list):
        print(f"\n[{i+1}/{len(subject_info_list)}] Processing {subject_info['subject_id']}...")
        
        results, error = process_single_subject(
            subject_info,  
            OUTPUT_DIR, 
            verbose=True
        )
        
        if results is not None:
            # Add subject ID to results
            for result in results:
                result['subject_id'] = subject_info['subject_id']
            all_results.extend(results)
            print(f"✓ Successfully processed {subject_info['subject_id']}")
        else:
            errors.append(error)
            print(f"✗ Failed to process {subject_info['subject_id']}")
    
    # Save combined results
    print(f"\n3. Saving combined results...")
    if all_results:
        combined_df = pd.DataFrame(all_results)
        combined_df.to_csv(OUTPUT_DIR / 'all_subjects_results.csv', index=False)
        
        # Summary statistics across all subjects
        print(f"\n4. Overall Summary:")
        print("="*80)
        print(f"Successfully processed: {len(subject_info_list) - len(errors)} subjects")
        print(f"Failed: {len(errors)} subjects")
        
        if errors:
            print(f"\nErrors:")
            for error in errors:
                print(f"  - {error}")
        
        # Overall performance metrics
        print(f"\nOverall Performance Metrics:")
        print("-" * 60)
        metrics = ['pearson r test', 'MSE test', 'r2 test']
        for metric in metrics:
            values = [r[metric] for r in all_results if not np.isnan(r[metric]) and not np.isinf(r[metric])]
            if values:
                print(f"{metric:<20}: mean={np.mean(values):>8.3f}, "
                      f"std={np.std(values):>8.3f}, "
                      f"min={np.min(values):>8.3f}, "
                      f"max={np.max(values):>8.3f}")
        
        # Channel performance analysis
        print(f"\nChannel Performance Analysis:")
        print("-" * 60)
        channel_performance = {}
        for result in all_results:
            ch_name = result['channel_name']
            if ch_name not in channel_performance:
                channel_performance[ch_name] = []
            channel_performance[ch_name].append(result['pearson r test'])
        
        # Show top performing channels across all subjects
        avg_performance = {ch: np.mean([r for r in perf if not np.isnan(r)]) 
                          for ch, perf in channel_performance.items() 
                          if len([r for r in perf if not np.isnan(r)]) > 0}
        
        top_channels = sorted(avg_performance.items(), key=lambda x: x[1], reverse=True)[:10]
        print("Top 10 channels across all subjects:")
        for i, (ch_name, avg_r) in enumerate(top_channels):
            print(f"  {i+1:2d}. {ch_name:<15}: r={avg_r:.3f}")
        
        print(f"\nResults saved to: {OUTPUT_DIR}")
        print(f"Individual subject results in subdirectories")
        print(f"Combined results: all_subjects_results.csv")
        
    else:
        print("No subjects were successfully processed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train individual regressors for all runs. EEG, ROI and MRI files are paired using subject and session information assuming BIDS-compliant filenames."
    )
    parser.add_argument("base_dir_eeg", type=str, help="Path to the EEG data directory, from eeg_loader.py.")
    parser.add_argument("base_dir_mri", type=str, help="Path to the fMRI data directory.")
    parser.add_argument("roi_dir", type=str, help="Path to the ROI definition directory containing per-subject ROI files.")
    parser.add_argument("out_dir", type=str, help="Directory where output will be saved.")

    args = parser.parse_args()
    # Fixed: Use the correct argument name (roi_dir instead of roi_path)
    main(args.base_dir_eeg, args.base_dir_mri, args.roi_dir, args.out_dir)