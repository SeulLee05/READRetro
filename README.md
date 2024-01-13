# READRetro: Natural Product Biosynthesis Planning with Retrieval-Augmented Dual-View Retrosynthesis
This is the official code repository for the paper [*READRetro: Natural Product Biosynthesis Planning with Retrieval-Augmented Dual-View Retrosynthesis (bioRxiv, 2023)*](https://www.biorxiv.org/content/10.1101/2023.03.21.533616v1).<br>
We also provide [a web version](https://readretro.net) for ease of use.

## Data
Download the necessary data folder `READRetro_data` from [Zenodo](https://zenodo.org/records/10495132) to ensure proper execution of the code and demonstrations in this repository.

The directory structure of `READRetro_data` is as follows:</br>

    READRetro_data
        ├── data.sh
        ├── data
        │   ├── model_train_data
        │   └── multistep_data
        ├── model
        │   ├── bionavi
        │   ├── g2s
        │   │   └── saved_models
        │   ├── megan
        │   └── retroformer
        │       └── saved_models
        ├── result
        └── scripts

Place `READRetro_data` into the READRetro directory (i.e., `READRetro/READRetro_data`) and run `sh data.sh` in `READRetro_data` to set up the data.</br>

Ensure the data is correctly located in `READRetro`. Verify the following:</br>
- `READRetro/retroformer/saved_models` should match `READRetro_data/model/retroformer/saved_models`.</br>
- `READRetro/g2s/saved_models` should match `READRetro_data/model/g2s/saved_models`.</br>
- `READRetro/data` should match `READRetro_data/data/multistep_data`.</br>
- `READRetro/result` should match `READRetro_data/result`.</br>
- `READRetro/scripts` should match `READRetro_data/scripts`.</br>

The directories `READRetro_data/model/bionavi`, `READRetro_data/model/megan`, and `READRetro_data/data/model_train_data` are required for reproducing the values in the manuscript.


## Installation
Run the following commands to install the dependencies:
```bash
conda create -n readretro python=3.8
conda activate readretro
conda install pytorch==1.12.0 cudatoolkit=11.3 -c pytorch
pip install easydict pandas tqdm numpy==1.22 OpenNMT-py==2.3.0 networkx==2.5
conda install -c conda-forge rdkit=2019.09
```

#### From pip
Alternatively, you can install the `readretro` package through pip:
```bash
conda create -n readretro python=3.8 -y
conda activate readretro
pip install readretro==1.1.0
```

## Model Preparation
We provide the trained models through Zenodo.<br>
You can use your own models trained using the official codes (https://github.com/coleygroup/Graph2SMILES and https://github.com/yuewan2/Retroformer).<br>
More detailed instructions can be found in `demo.ipynb`.

## Single-step Planning and Evaluation
Run the following commands to evaluate the single-step performance of the models:
```bash
CUDA_VISIBLE_DEVICES=${gpu_id} python eval_single.py                    # ensemble
CUDA_VISIBLE_DEVICES=${gpu_id} python eval_single.py -m retroformer     # Retroformer
CUDA_VISIBLE_DEVICES=${gpu_id} python eval_single.py -m g2s -s 200      # Graph2SMILES
```

## Multi-step Planning
Run the following command to plan paths of multiple products using multiprocessing:
```bash
CUDA_VISIBLE_DEVICES=${gpu_id} python run_mp.py
# e.g., CUDA_VISIBLE_DEVICES=0 python run_mp.py
```
You can modify other hyperparameters described in `run_mp.py`.<br>
Lower `num_threads` if you run out of GPU capacity.

Run the following command to plan the retrosynthesis path of your own molecule:
```bash
CUDA_VISIBLE_DEVICES=${gpu_id} python run.py ${product}
# e.g., CUDA_VISIBLE_DEVICES=0 python run.py 'O=C1C=C2C=CC(O)CC2O1'
```
#### Using a command from pip
``` bash
run_readretro -rc ${retroformer_ckpt} -gc ${g2s_ckpt} ${product}
# e.g., run_readretro -rc retroformer/saved_models/biochem.pt -gc g2s/saved_models/biochem.pt 'O=C1C=C2C=CC(O)CC2O1'
# you can replace the checkpoints with your own trained checkpoints of retroformer and g2s
```
You can modify other hyperparameters described in `run.py`.

## Multi-step Evaluation
Run the following command to evaluate the planned paths of the test molecules:
```bash
python eval.py ${save_file}
# e.g., python eval.py result/debug.txt
```

## Demo
You can reproduce the figures and tables presented in the paper or train your own models by utilizing the provided `demo.ipynb`.

<!-- 
## Citation
If you find this repository and our paper useful, we kindly request to cite our work.

```BibTex
@article{lee2023READRetro,
  author    = {Seul Lee and
               Taein Kim and
               Min-Soo Choi and
               Yejin Kwak and
               Jeongbin Park and
               Sung Ju Hwang and
               Sang-Gyu Kim},
  title     = {READRetro: Natural Product Biosynthesis Planning
               with Retrieval-Augmented Dual-View Retrosynthesis},
  journal   = {bioRxiv},
  year      = {2023}
}
``` -->
