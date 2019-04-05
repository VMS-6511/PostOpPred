# PostOpPred
Predicting Post-Operative Survival of Lung Cancer Patients

## Environment Instructions

Run conda create -n PostOp python=3.6

Run source activate PostOp

Run pip install -r requirements.txt

### For Notebooks
Run python -m ipykernel install --user --name postop --display-name "PostOp"

Run jupyter notebook

### Python Files
Run python preprocessing/preprocessing.py

Run python train/neural_network.py