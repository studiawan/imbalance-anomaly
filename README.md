# imbalance-anomaly
Evaluation of anomaly detection in imbalanced authentication logs.

## Creating a new virtual environment

1. Create a conda virtual environment

   `conda create --name imbalance-anomaly python=3.6`

2. Activate the environment

   `conda activate imbalance-anomaly`

## Cloning the repository and install the package

1. Clone this repository

   `git clone https://github.com/studiawan/imbalance-anomaly.git`

2. Go to the project directory. The rest of the instructions run in this directory
    
   `cd imbalance-anomaly`

3. In `setup.py` file, change `tensorflow-gpu` to `tensorflow` if you do not run on a GPU

4. Install this package in the activated virtual environment
   
   `pip install -e .`
   

## Preparing datasets and Glove embedding

1. Copy the datasets from [`imbalance-anomaly-gt`](https://github.com/studiawan/imbalance-anomaly-gt) repository to directory `imbalance-anomaly/datasets`. It is assumed that the directory of both repository are located in `~/Git`. Please change according to your own directory structure 
   
   `cp ~/Git/imbalance-anomaly-gt/datasets/casper-rw/log.all.pickle ~/Git/imbalance-anomaly/datasets/casper-rw/log.all.pickle`
   
   `cp ~/Git/imbalance-anomaly-gt/datasets/dfrws-2009/log.all.pickle ~/Git/imbalance-anomaly/datasets/dfrws-2009/log.all.pickle`
   
   `cp ~/Git/imbalance-anomaly-gt/datasets/honeynet-challenge7/log.all.pickle ~/Git/imbalance-anomaly/datasets/honeynet-challenge7/log.all.pickle`

2. Extract the Glove pre-trained embedding

   `tar -xzvf glove/glove6B.50d.tar.gz --directory glove/`

## Running the experiment
   
1. Run experiments for all methods from Keras library. Type dataset name and method name after the script. The supported datasets are `casper-rw`, `dfrws-2009`, and `honeynet-challenge7`. The supported methods are `lstm` and `cnn`.

   Command:

   `python imbalance_anomaly/experiment/experiment_keras.py dataset_name method_name`
   
   Example:
   
   `python imbalance_anomaly/experiment/experiment_keras.py casper-rw lstm`

2. The experimental results are located in `imbalance_anomaly/datasets/$DATASET_NAME$/` where `$DATASET_NAME$` is one of the datasets: `casper-rw`, `dfrws-2009`, and `honeynet-challenge7`. The file name format for experimental results is `$METHOD_NAME$.evaluation.csv`.

3. Pretty print the csv file of experimental results
   
   `column -s, -t datasets/$DATASET_NAME$/$METHOD_NAME$.evaluation.csv`