<!-- <div align="center"> -->
# SurfDeign

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>

[//]: # ([![Paper]&#40;http://img.shields.io/badge/ICML_2023-arXiv.2302.01649-B31B1B.svg&#41;]&#40;https://arxiv.org/abs/2302.01649&#41;)
[//]: # (<!-- [![Conference]&#40;http://img.shields.io/badge/AnyConference-year-4b44ce.svg&#41;]&#40;https://papers.nips.cc/paper/2020&#41; -->)
<!-- </div> -->
<!-- ## Description -->

## Installation  
```bash
# create conda virtual environment
env_name=SurfDesign

conda create -n ${env_name} python=3.7 pip
conda activate ${env_name}
bash install.sh
```
todo: version is old, improve version 

[//]: # (## Structure-based protein sequence design &#40;inverse folding&#41;)
[//]: # (**Pretrained model weights** &#40;[Zenodo]&#40;https://zenodo.org/records/10046338&#41;&#41;)
[//]: # (| model                | training data | checkpoint |)
[//]: # (|----------------------|------------|------------|)
[//]: # (| `protein_mpnn_cmlm`    | cath_4.2   | [link]&#40;https://zenodo.org/records/10046338/files/protein_mpnn_cmlm.zip?download=1&#41;       |)
[//]: # (| `lm_design_esm2_650m`  | cath_4.2   | [link]&#40;https://zenodo.org/records/10046338/files/lm_design_esm2_650m.zip?download=1&#41;       |)
[//]: # (| `lm_design_esm2_650m`  | multichain   | [link]&#40;https://zenodo.org/records/10046338/files/lm_design_esm2_650m_multichain.zip?download=1&#41;       |)


### Data
**Download the preproceesd CATH datasets**
- CATH 4.2 dataset provided by [Generative Models for Graph-Based Protein Design (Ingraham et al, NeurIPS'19)](https://papers.nips.cc/paper/2019/hash/f3a4ff4839c56a5f460c88cce3666a2b-Abstract.html)
- CATH 4.3 dataset provided by [Learning inverse folding from millions of predicted structures (Hsu et al, ICML'22)](https://www.biorxiv.org/content/10.1101/2022.04.10.487779v1) 
```bash
bash scripts/download_cath.sh
```
Go check `configs/datamodule/cath_4.*.yaml` and set `data_dir` to the path of the downloaded CATH data. 

**Dowload PDB complex data (multichain)**

This dataset curated protein (multichain) complexies from Protein Data Bank (PDB). 
It is provided by [Robust deep learning-based protein sequence design using ProteinMPNN](https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1). 
See their [github page](https://github.com/dauparas/ProteinMPNN/blob/main/training/README.md) for more details.
```bash
bash scripts/download_multichain.sh
```
Go check `configs/datamodule/multichain.yaml` and set `data_dir` to the path of the downloaded multichain data. 

**Surface Generation and Preprocess**

Use https://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set_splits.json to download the pdb files and preprocess them. 

[//]: # (Using `preprocess.py` to convert CATH data to PDB format.)


### Training
In the following sections, we will use CATH 4.2 dataset as an runing example. 
You can likewise build your models on the multichain dataset to accommodate protein complexies.


#### Example 1: Non-autoregressive (NAR) ProteinMPNN baseline

Training NAR ProteinMPNN with conditional masked language modeling (CMLM)

```bash
export CUDA_VISIBLE_DEVICES=0
# or use multi-gpu training when you want:
# export CUDA_VISIBLE_DEVICES=0,1

exp=fixedbb/protein_mpnn_cmlm  
dataset=cath_4.2
name=fixedbb/${dataset}/protein_mpnn_cmlm

python train.py experiment=${exp} datamodule=${dataset} name=${name} logger=tensorboard trainer=ddp_fp16 
```

Some flags for training:

| Argument              | Usage                                                                                                                   |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------|
| `experiment`          | experiment config. see `configs/experiment/` folder                                                                     |
| `datamodule`          | dataset config. see `configs/datamodule` folder                                                                         |
| `name`                | experiment name, deciding the directory path your experiment saving to, e.g., `xxx/run/logs/${name}`                    |
| `logger`              | config of which ml experiment logger to use, e.g., tensorboard.                                                         |
| `train.force_restart` | set to `true` to force retrain the experiment under `${name}`. otherwise will resume training from the last checkpoint. |


#### Example 2: SurfDesign

Training <span style="font-variant:small-caps;">SurfDesign</span> upon ESM-2 650M.
> Training would take approxmiately 6 hours on one A100 GPU. 
```bash
dataset=cath_4.2
# change dataset 
dataset=cath_4.3
exp=fixedbb/surf_design_esm2_650m
name=fixedbb/${dataset}/surf_design_esm2_650m

CUDA_VISIBLE_DEVICES=2 python train.py experiment=${exp} datamodule=${dataset} name=${name} logger=tensorboard trainer=ddp_fp16 
```
Building <span style="font-variant:small-caps;">LM-Design</span> upon ESM-2 series using `exp=fixedbb/surf_design_esm2`. Please check `SurfDesign/configs/experiment/fixedbb`.


### Evaluation/inference on valid/test datasets
```bash
dataset=cath_4.2
# change dataset
dataset=cath_4.3 

name=fixedbb/${dataset}/surf_design_esm2_650m
exp_path=/ai/design/SurfDesign/logs/${name}

# compute aar 
CUDA_VISIBLE_DEVICES=1 python test.py experiment_path=${exp_path} data_split=test ckpt_path=best.ckpt task.generator.max_iter=1
mode=predict 

# compute ppl
mode=test 

# compute subset test 
datamodule.chain_set_splits_json=test_split_L100.json   # length < 100
datamodule.chain_set_splits_json=test_split_sc.json   # single chain
```

Some flags for generation

| Argument                             | Usage                                                                                                            |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `experiment_path`                    | folder that saves experiment (.hydra, checkpoints, tensorboard, etc)                                             |
| `data_split`                         | `valid` or `test` dataset.                                                                                       |
| `mode`                               | `predict` for generating sequence & calculating amino acid sequence recovery; `test` for evaluation for nll, ppl |
|  | |
| `task.generator`                     | arguments for sequence generator/sampler                                                                         |
| - `max_iter=<int>`                   | maximum decoding iteration (default: `5` for LM-Design, `1` for ProtMPNN-CMLM)                                   |
| - `strategy=[denoise, mask_predict]` | decoding strategy. (default: `denoise` for LM-Design, `mask_predict` for ProtMPNN-CMLM)                          |
| - `temperature=<float>`              | temperature for sampling. set to 0 to disable for deterministic sampling (default: `0`)                          |
| - `eval_sc=<bool>`                   | additional evaluating scTM score using ESMFold. (default: `false`)                                               |

### Designing sequences from a pdb file using a trained model in Notebook

**Example 1: ProteinMPNN-CMLM**

```python
from src.utils import compose_config as Cfg
from src.tasks import Designer

# 1. instantialize designer
exp_path = "xxx/run/logs/fixedbb/cath_4.2/protein_mpnn_cmlm"
cfg = Cfg(cuda=True, generator=Cfg(max_iter=1, strategy='mask_predict', temperature=0, eval_sc=False, ))
designer = Designer(experiment_path=exp_path, cfg=cfg)

# 2. load structure from pdb file
pdb_path = "xxx/data/3uat_variants/3uat_GK.pdb"
designer.set_structure(pdb_path)

# 3. generate sequence from the given structure
designer.generate()

# 4. calculate evaluation metircs
designer.calculate_metrics()  ## prediction: SSYNPPILLLGPFAEELEEELVEENPERAGRPVPFTTEPPSPDETEGETYLYISSLEEAEELIESNRFLEAGEENNELVGISLEAIRSVARAGKLAILDTGGEAVEKLEEANIEPIVIFLVPKSVEDVRRVFPDLTEEEAEELTSEDEELLEEFKELLDAVVSGSTLEEVLEEIREVIEEASS
## recovery: 0.37158469945355194
```


**Example 2: <span style="font-variant:small-caps;">SurfDesign</span>**

```python
from src.utils import compose_config as Cfg
from src.tasks import Designer

# 1. instantialize designer
exp_path = "xxx/run/logs/fixedbb/cath_4.2/surf_design_esm2_650m"
cfg = Cfg(cuda=True, generator=Cfg(max_iter=5, strategy='denoise', temperature=0, eval_sc=False, ))
designer = Designer(experiment_path=exp_path, cfg=cfg)  # TODO: ....... important!

# 2. load structure from pdb file
pdb_path = "xxx.pdb"
designer.set_structure(pdb_path)

# 3. generate sequence from the given structure
designer.generate()
# you can override generator arguments by passing generator_args, e.g.,
designer.generate(generator_args={'max_iter': 5, 'temperature': 0.1, })

# 4. calculate evaluation metircs
designer.calculate_metrics()  ## prediction: LNYTRPVIILGPFKDRMNDDLLSEMPDKFGSCVPHTTRPKREYEIDGRDYHFVSSREEMEKDIQNHEFIEAGEYNDNLYGTSIESVREVAMEGKHCILDVSGNAIQRLIKADLYPIAIFIRPRSVENVREMNKRLTEEQAKEIFERAQELEEEFMKYFTAIVEGDTFEEIYNQVKSIIEEESG
## recovery: 0.7595628415300546
```


## Acknowledgements
SurfDesign extends its gratitude to the following projects and individuals:

- [PyTorch Lightning](https://www.pytorchlightning.ai/) and [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) for providing a robust foundation for our development process.

SurfDesign draws inspiration and leverages/modifies implementations from the following repositories:
- [jingraham/neurips19-graph-protein-design](https://github.com/jingraham/neurips19-graph-protein-design) for the preprocessed CATH dataset and data pipeline implementation.
- [facebook/esm](https://github.com/facebookresearch/esm/) for their ESM implementations, pretrained model weights, and data pipeline components like `Alphabet`.
- [dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN/) for the ProteinMPNN implementation and multi-chain dataset.
- [A4Bio/PiFold](https://github.com/A4Bio/PiFold) for their PiFold implementation.
- [jasonkyuyim/se3_diffusion](https://github.com/jasonkyuyim/se3_diffusion) for their self-consistency structural evaluation implementation.
- ByProt
- SurfPro

We express our sincere appreciation to the authors of these repositories for their invaluable contributions to the development of SurfDesign.
