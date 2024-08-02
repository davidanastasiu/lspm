## LSPM

We present `LSPM`, the multivariate time series prediction model we detail in our paper, "Long-Term Hydrologic Time Series Prediction with LSPM" which will be presented at CIKM 2024. If you make use of our code or data, please cite our paper.

```bibtex
@inproceedings{ZhouA2024,
    author      = {Sicheng Zhou and David C. Anastasiu},
    title       = {Long-Term Hydrologic Time Series Prediction with LSPM},
    booktitle   = {Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
    series      = {CIKM'24},
    year        = {2024},
    location    = {Boise, ID, USA},
    pages       = {},
    numpages    = {5},
    publisher   = {ACM},
    address     = {New York, NY, USA},
    url         = {https://doi.org/10.1145/3627673.3679957},
    doi         = {10.1145/3627673.3679957},
}
```

## Preliminaries

Experiments were executed in an Anaconda 3 environment with Python 3.8.3. The following will create an Anaconda environment and install the requisite packages for the project.

```bash
conda create -n LSPM python=3.8.8
conda activate LSPM
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch
python -m pip install -r requirements.txt
```

## Files organization

Download the datasets from [here](https://clp.engr.scu.edu/static/datasets/seed_datasets.zip) and upzip the files in the data_provider directory. In the ./data_provider/datasets directory, there should now be 4 stream sensor (file names end with _S_fixed.csv) and 4 rain sensor (file names end with _R_fixed.csv) datasets.

## Parameters setting

--stream_sensor: stream dataset file name. The file should be csv file.

--rain_sensor: rain dataset file name. The file should be csv file.

--train_volume: train set size.

--hidden_dim: hidden dim of lstm layers.

--cnn_dim: hidden dim of cnn layers.

--layer: number of layers.

--model: model name, used to generate the pt file and predicted file names.

--mode: set it to 'train' or 'inference' with an existing pt_file.

--pt_file: if set, the model will be loaded from this pt file, otherwise check the file according to the assigned parameters.

--save: if save the predicted file of testset, set to 1, else 0.

--outf: default value is './output', the model will be saved in the train folder in this directory.

Refer to the annotations in `run.py` for other parameter settings. Default parameters for reproducing are set in the files (file names start with opt and end with .txt) under './models/'.

## Training and Inferencing

Execute the Jupyter notebook experiments.ipynb to train models and conduct inferences on the test sets of the four stream datasets described in the associated paper.

The Jupyter notebook example.ipynb shows how to train a model via command line commands and use specific model functions to perform inference on the SFC sensor dataset.

