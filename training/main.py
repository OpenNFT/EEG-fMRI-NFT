import sys
import tqdm
from pathlib import Path
from utils import find_subject_files, load_subject_data, save_subject_results, print_results_summary, print_overall_summary
from train import train_efp_model

def main():
    # Check for correct number of command-line arguments
    if len(sys.argv) != 5:
        print("Usage: python main.py <eeg_dir> <mri_dir> <roi_dir> <output_dir>")
        print("Example: python main.py Z:\\RR_MH_with_parser\\eeg_windows Z:\\RR_MH_with_parser\\mri Z:\\RR_MH_with_parser\\mri\\ROI Z:\\RR_MH_with_parser\\trained_regressors")
        sys.exit(1)

    # Get directory paths from command-line arguments
    base_dir_eeg = sys.argv[1]
    base_dir_mri = sys.argv[2]
    roi_dir = sys.argv[3]
    output_dir = sys.argv[4]

    # Validate directory paths
    for dir_path in [base_dir_eeg, base_dir_mri, roi_dir, output_dir]:
        if not Path(dir_path).exists():
            print(f"Error: Directory does not exist: {dir_path}")
            sys.exit(1)

    # Find subject files
    subject_info_list = find_subject_files(base_dir_eeg, base_dir_mri, roi_dir)
    if not subject_info_list:
        print("No valid subjects found. Exiting.")
        sys.exit(1)

    all_results = []
    errors = []

    # Process each subject
    for subject_info in tqdm.tqdm(subject_info_list, desc="Processing subjects"):
        subject_id = subject_info['subject_id']
        
        # Validate subject_id format
        parts = subject_id.split('_')
        if len(parts) < 2:
            errors.append(f"Invalid subject_id format for {subject_id}. Expected sub-xxx_ses-xxx[_run-xxx]")
            continue

        # Load subject data
        data_result, error = load_subject_data(subject_info)
        if error or data_result is None:
            errors.append(error)
            continue

        eeg_features, targets, enhanced_metadata = data_result
        n_channels = enhanced_metadata['n_channels']
        n_bands = enhanced_metadata['n_bands']
        n_timepoints = enhanced_metadata['n_timepoints']
        channel_names = enhanced_metadata['channel_names']

        # Train model
        try:
            results, models = train_efp_model(
                data=eeg_features,
                target=targets,
                n_channels=n_channels,
                n_bands=n_bands,
                n_timepoints=n_timepoints,
                channel_names=channel_names,
                verbose=True
            )
        except Exception as e:
            errors.append(f"Training failed for {subject_id}: {str(e)}")
            continue

        # Save results with subject-specific subdirectory
        subject_output_dir = Path(output_dir) / subject_id
        try:
            save_subject_results(results, subject_id, subject_output_dir, models, enhanced_metadata)
        except Exception as e:
            errors.append(f"Saving failed for {subject_id}: {str(e)}")
            continue

        all_results.extend(results)
        print_results_summary(subject_id, results)

    # Print overall summary
    print_overall_summary(all_results, subject_info_list, errors)

if __name__ == "__main__":
    main()