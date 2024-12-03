# Changelog for llama embeddings

In the base implementation, roBERTa (loaded via huggingface) is used to generate word embeddings. We are replacing roBERTa with llama-2-7b (loaded via huggingface as well) to create higher-dimensional embeddings without changing the structure of the model. 

## Changes

1.	nrms_ebnerd.py

-	imported AutoModelForCausalLM from transformer (huggingface’s library)
-	changed transformer_model_name to “meta-llama/Llama-2-7b-hf”
o	adjusted how the models are built (dependant on the model classes in hf)

2.	_articles.py

-	adjusted convert_text2encoding_with_transformers function
o	max_length needs to be initialized 
o	special tokens set to TRUE
o	padding is set to max length

3.	smaller things

in venv/lib/python3.11/site-packages/ebrec/utils/_nlp.py

def get_transformers_word_embeddings(model: AutoModel):
    return model.model.embed_tokens.weight.data.to("cpu").numpy()


## Outcomes

The NRMS is initialized with the llama embeddings (still saved in word2vec_embeddings). 



# Introduction
Hello there 👋🏽

We recommend to check the repository frequently, as we are updating and documenting it along the way!

## EBNeRD 
Ekstra Bladet Recommender System repository, created for the RecSys'24 Challenge. 

# Getting Started
We recommend [conda](https://docs.conda.io/projects/conda/en/latest/glossary.html#conda-environment) for environment management, and [VS Code](https://code.visualstudio.com/) for development. To install the necessart packages and run the example notebook:

```
# 1. Create and activate a new conda environment
conda create -n <environment_name> python=3.11
conda activate <environment_name>

# 2. Clone this repo within VSCode or using command line:
git clone https://github.com/ebanalyse/ebnerd-benchmark.git

# 3. Install the core ebrec package to the enviroment:
pip install .
```

We have experienced issues installing *tensorflow* for M1 Macbooks (```sys_platform == 'darwin'```) when using conda. To avoid this, we suggest to use venv if running on macbooks.
```
python3 -m .venv .venv
source  .venv/bin/activate
```

Installing ```.venv``` in project folder:
```
conda create -p .venv python==3.11.8
conda activate ./.venv
```

## Running GPU
```
tensorflow-gpu; sys_platform == 'linux'
tensorflow-macos; sys_platform == 'darwin'
```

# Algorithms
To get started quickly, we have implemented a couple of News Recommender Systems, specifically, 
[Neural Recommendation with Long- and Short-term User Representations](https://aclanthology.org/P19-1033/) (LSTUR),
[Neural Recommendation with Personalized Attention](https://arxiv.org/abs/1907.05559) (NPA),
[Neural Recommendation with Attentive Multi-View Learning](https://arxiv.org/abs/1907.05576) (NAML), and
[Neural Recommendation with Multi-Head Self-Attention](https://aclanthology.org/D19-1671/) (NRMS). 
The source code originates from the brilliant RS repository, [recommenders](https://github.com/recommenders-team/recommenders). We have simply stripped it of all non-model-related code.


# Notebooks
To help you get started, we have created a few notebooks. These are somewhat simple and designed to get you started. We do plan to have more at a later stage, such as reproducible model trainings.
The notebooks were made on macOS, and you might need to perform small modifications to have them running on your system.

## Model training
We have created a [notebook](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/00_quick_start/nrms_ebnerd.ipynb) where we train NRMS on EB-NeRD - this is a very simple version using the demo dataset.

## Data manipulation and enrichment
In the [dataset_ebnerd](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/00_quick_start/dataset_ebnerd.ipynb) demo, we show how one can join histories and create binary labels.

# Reproduce EB-NeRD Experiments

Activate your enviroment:
```
conda activate <environment_name>
```

### [NRMSModel](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/src/ebrec/models/newsrec/nrms.py) 

```
python examples/reproducibility_scripts/ebnerd_nrms.py
  --datasplit ebnerd_small \
  --epochs 5 \
  --bs_train 32 \
  --bs_test 32 \
  --history_size 20 \
  --npratio 4 \
  --transformer_model_name FacebookAI/xlm-roberta-large \
  --max_title_length 30 \
  --head_num 20 \
  --head_dim 20 \
  --attention_hidden_dim 200 \
  --learning_rate 1e-4 \
  --dropout 0.20
```

Tensorboards:
```
tensorboard --logdir=ebnerd_predictions/runs
```

### [NRMSDocVec](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/src/ebrec/models/newsrec/nrms_docvec.py) 

```
python examples/reproducibility_scripts/ebnerd_nrms_docvec.py \
  --datasplit ebnerd_small \
  --epochs 5 \
  --bs_train 32 \
  --history_size 20 \
  --npratio 4 \
  --document_embeddings Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet \
  --head_num 16 \
  --head_dim 16 \
  --attention_hidden_dim 200 \
  --newsencoder_units_per_layer 512 512 512 \
  --learning_rate 1e-4 \
  --dropout 0.2 \
  --newsencoder_l2_regularization 1e-4
```

Tensorboards:
```
tensorboard --logdir=ebnerd_predictions/runs
```



