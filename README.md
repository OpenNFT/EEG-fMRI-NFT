# EEG-fMRI-NFT

A FAST API implementation for EEG-fMRI neurofeedback. Receives data from a LSL stream in real time and returns predictions on request using a sklearn predictor. 

## Requirements
To use this repository, you will need the ``scipy``, ``scikit-learn`` (make sure you are using the same version used to train your model), ``fastapi``, ``pydantic`` and ``stockwell`` packages.

In anaconda:
```
conda create -n myenv scipy scikit-learn=1.5.2 fastapi pydantic stockwell -c conda-forge -y
```

Or using pip:

```
pip install scipy scikit-learn fastapi pydantic stockwell
```

While we do provide a Dockerfile, be advised that connecting to an LSL stream from a Docker container is not trivial, depending on your system and OS.
## Configuration
The information specific to your setting is stored in the ``model`` directory.
Replace ``model/estimator.joblib`` with your estimator, and ``model/bands_info.json`` with the configuration file generated during training.
The channel selected for inference and the name of the LSL stream are specified respectively in ``channel.txt`` and ``stream_name.txt``.

## Training

Please refer to the instructions in the ``training`` directory.

## Usage
You can use the following command:
```
python -m uvicorn main:app --reload
```
