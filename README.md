# FAIR-Tuning
## Prerequisities
### Environment
- docker version: nvcr.io/nvidia/pytorch:23.03-py3
- python: 3.8.10
- requirements.txt `pip install -r requirements.txt`

### Folder Structure
```
FAIR-Tuning/
├── embedding/ 
│   └── xxx.pt
├── clinical_information/
│   └── xxx_clinical_information.pkl
├── pretrained_weight/
│   └── race
│   └── gender
├── datasets.py
├── fairmetric.py
├── inference.py
├── main.py
├── network.py
├── requirements.txt
├── run_inference.sh
├── run_train.sh
└── README.md
```
## Usage
### Dataset
- Prepare image embeddings and put them under `embedding/` folder.
    - e.g. `sudo mount <ip>:<path_to_remote_embeddng> <path_to_local_embedding>` (suggest to download the whole embedding files)
- Prepare clinical information pickle files and put them under `clinical_information/` folder.

### Training
```
bash run_train.sh
```
- Add a `model/` folder to store models.
- Set training argument in `run_train.sh`.
    - `SENSITIVE`: the demographic subgroups for mitigating bias
    - `--model_path`: path to store the training model
    - `--task`: 1 for cancer classification, 2 for tumor detection, 3 for survival analysis, 4 for genetic mutation classification
    - `--reweight`: add this argument to do the FAIR-Tuning stage.
    - other argument details can be seen in the `main.py`.

### Inference
```
bash run_inference.sh
```
- Set inference argument in `run_inference.sh`.
    - `SENSITIVE`: the demographic subgroups for mitigating bias
    - `--weight_path`: path to the existed weight
    - `--task`: 1 for cancer classification, 2 for tumor detection, 3 for survival analysis, 4 for genetic mutation classification
    - `--reweight`: add this argument to select the weight after FAIR-Tuning.
    - other argument details can be seen in the `inference.py`.
