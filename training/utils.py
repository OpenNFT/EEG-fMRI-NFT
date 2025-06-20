import numpy as np
import nibabel as nib
import pandas as pd
import json
import glob
import joblib
import pickle  # Added missing import
from pathlib import Path
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import tqdm

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
    """Generate regularization grid for Ridge regression."""
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

def grid_search_with_alpha_grid(channel, estimator, inner_splitter, metric, regularization_grid_search, 
                                train_data, train_targets, n_bands, n_timepoints, verbose=0):
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

# ========== File Finding Functions ==========
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
            
            # Look for fMRI files in the mri directory with .nii.gz patterns only
            fmri_patterns = [
                f"{sub_part}_{ses_part}_task-V1LOC_acq-3_dir-AP_bold_fugue_ds_st_mc.nii.gz",
                f"{sub_part}_{ses_part}_task-V1LOC_*_bold*.nii.gz",
                f"{sub_part}_{ses_part}_*_bold*.nii.gz",
                f"{sub_part}_{ses_part}_*.nii.gz",
                f"{sub_part}_*_bold*.nii.gz"
            ]
            
            fmri_file = None
            for pattern in fmri_patterns:
                fmri_files = list(fmri_base.glob(pattern))
                if fmri_files:
                    fmri_file = fmri_files[0]  # Take first match
                    break
            
            if not fmri_file:
                print(f"Warning: fMRI file not found for {subj_name}")
                continue
            
            # Look for ROI files in the roi directory (.nii.gz only)
            roi_patterns = [
                f"{sub_part}_{ses_part}_roi.nii.gz",
                f"{sub_part}_*_roi.nii.gz"
            ]
            
            roi_file = None
            for pattern in roi_patterns:
                roi_files = list(roi_base.glob(pattern))
                if roi_files:
                    roi_file = roi_files[0]  # Take first match
                    break
            
            if not roi_file:
                print(f"Warning: ROI file not found for {subj_name}")
                continue
                
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

def load_subject_data(subject_info):
    """Load and prepare data for a single subject."""
    subject_id = subject_info['subject_id']
    eeg_file = subject_info['eeg_file']
    fmri_file = subject_info['fmri_file']
    roi_file = subject_info['roi_file']
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
        
        # Load fMRI and ROI data
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

# ========== Data Processing Functions ==========
def process_eeg_data_shape(eeg_data, metadata, channel_names):
    """Process EEG data and extract shape information."""
    n_bands = metadata.get('n_bands', 10)
    
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
        raise ValueError(f"Unsupported EEG data format: {eeg_data.shape}")
    
    return eeg_features, n_channels, n_bands, n_timepoints, n_windows_eeg

def validate_channel_names(channel_names, n_channels):
    """Validate and fix channel names if needed."""
    if channel_names and len(channel_names) != n_channels:
        print(f"Warning: Channel names count ({len(channel_names)}) doesn't match n_channels ({n_channels})")
        if len(channel_names) > n_channels:
            channel_names = channel_names[:n_channels]
        else:
            # Pad with default names
            channel_names.extend([f"Channel_{i}" for i in range(len(channel_names), n_channels)])
    
    return channel_names

def create_fmri_target(fmri_file, roi_file, n_windows_eeg):
    """Create target variable from fMRI and ROI data."""
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
        # Interpolate if fMRI has fewer timepoints
        interp_func = interp1d(np.linspace(0, 1, len(z_targets)), z_targets, kind='linear')
        trimmed_targets = interp_func(np.linspace(0, 1, n_windows_eeg))
    else:
        trimmed_targets = z_targets
    
    return trimmed_targets

# ========== File Saving Functions ==========
def save_subject_results(results, subject_id, subject_output_dir, models=None, enhanced_metadata=None):
    """Save results for a single subject."""
    # Extract subject parts from subject ID
    parts = subject_id.split('_')
    if len(parts) >= 2:
        sub_part = parts[0]  # e.g., "sub-050"
        ses_part = parts[1]  # e.g., "ses-day2"
    else:
        sub_part = subject_id
        ses_part = "unknown"
    
    subject_output_dir = Path(subject_output_dir)
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results as DataFrame with sub-session ID in filename
    results_df = pd.DataFrame(results)
    # Add sub_id and ses_id columns to the results
    results_df['sub_id'] = sub_part
    results_df['ses_id'] = ses_part
    results_df.to_csv(subject_output_dir / f'{sub_part}_{ses_part}_results.csv', index=False)
    
    if models is not None:
        # Save individual models using joblib with subject and session info
        models_dir = subject_output_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        print(f"Saving {len(models)} models using joblib...")
        channel_names = enhanced_metadata.get('channel_names') if enhanced_metadata else None
        
        for i, model in enumerate(models):
            channel_name = channel_names[i] if channel_names and i < len(channel_names) else f"Channel_{i}"
            # Clean channel name for filename (remove special characters)
            clean_channel_name = "".join(c for c in channel_name if c.isalnum() or c in ('_', '-'))
            # Create filename with subject, session, and channel info
            model_filename = f"{sub_part}_{ses_part}_channel_{i:03d}_{clean_channel_name}.joblib"
            model_path = models_dir / model_filename
            joblib.dump(model, model_path)
            
        # Save model metadata as joblib file with sub-session ID
        model_info = {
            'subject_id': subject_id,
            'sub_id': sub_part,
            'ses_id': ses_part,
            'channel_names': channel_names,
            'model_files': [f"{sub_part}_{ses_part}_channel_{i:03d}_{(''.join(c for c in channel_names[i] if c.isalnum() or c in ('_', '-'))) if channel_names and i < len(channel_names) else f'Channel_{i}'}.joblib" 
                           for i in range(len(models))]
        }
        
        if enhanced_metadata:
            model_info.update({
                'n_channels': enhanced_metadata['n_channels'],
                'n_bands': enhanced_metadata['n_bands'],
                'n_timepoints': enhanced_metadata['n_timepoints']
            })
        
        # Save model info as joblib with sub-session ID in filename
        joblib.dump(model_info, subject_output_dir / f'{sub_part}_{ses_part}_models_info.joblib')
        
        # Also save as JSON for easy reading with sub-session ID in filename
        model_info_json = model_info.copy()
        with open(subject_output_dir / f'{sub_part}_{ses_part}_models_info.json', 'w') as f:
            json.dump(model_info_json, f, indent=2)
        
        # Save the main models file in the format you want: sub-050_ses-day2_models.joblib
        joblib.dump(model_info, subject_output_dir / f'{sub_part}_{ses_part}_models.joblib')
        
        # Also save as pickle for compatibility
        with open(subject_output_dir / f'{sub_part}_{ses_part}_models.pkl', 'wb') as f:
            pickle.dump(model_info, f)
    
    if enhanced_metadata is not None:
        # Save enhanced metadata with sub-session ID in filename
        enhanced_metadata_copy = enhanced_metadata.copy()
        enhanced_metadata_copy['sub_id'] = sub_part
        enhanced_metadata_copy['ses_id'] = ses_part
        
        # Save enhanced metadata as joblib with sub-session ID in filename
        joblib.dump(enhanced_metadata_copy, subject_output_dir / f'{sub_part}_{ses_part}_metadata.joblib')
        
        # Also save as JSON for easy reading with sub-session ID in filename
        enhanced_metadata_json = enhanced_metadata_copy.copy()
        with open(subject_output_dir / f'{sub_part}_{ses_part}_metadata.json', 'w') as f:
            json.dump(enhanced_metadata_json, f, indent=2)

def print_results_summary(subject_id, results):
    """Print summary of results for a subject."""
    print(f"\nResults Summary for {subject_id}:")
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

def print_overall_summary(all_results, subject_info_list, errors):
    """Print overall summary across all subjects."""
    print(f"\nOverall Summary:")
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