# Facial Recognition System Alpha Prototype

## Setup and Installation

Download the following dependencies:
```bash
cd alpha-prototype

conda create -n alpha-prototype python=3.10
conda activate alpha-prototype
pip install requests tqdm

git clone https://github.com/serengil/deepface.git
cd deepface
pip install -e .
```

## Project Structure
- `download.py`: Downloads and extracts the face dataset
- `experiment_maker.py`: Creates template and probe datasets from reference data
- `alpha_prototype.py`: Core facial recognition system
- `experiment_runner.py`: Tests the system's performance

## Usage Instructions (ensure you are in the alpha-prototype directory)

### 1. Download the dataset
```bash
python download.py
```

### 2. Create experimental datasets
```bash
python experiment_maker.py
```

### 3. Run the facial recognition system (optional, it is used for sanity check)
```bash
python alpha_prototype.py 
```

### 4. Evaluate system performance
```bash
python experiment_runner.py
```

## Parameters

You can adjust various parameters in each module:

- `experiment_maker.py`: Control the number of individuals and images per individual
- `alpha_prototype.py`: Change the recognition model (VGG-Face, Facenet, etc.) and distance metric
- `experiment_runner.py`: Set thresholds for accepting matches

## Notes
- The system assumes a closed-world scenario where all probe subjects are in the template database
- Performance metrics are saved in the results directory for analysis
