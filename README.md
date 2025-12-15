Final Project for Deep Learning Course

This repository contains the code for a deep learning project on **detecting hate and abusive language in Indonesian tweets** using:

- **IndoBERT** as the base language model
- **Unsupervised SimCSE** for contrastive pretraining of sentence embeddings
- A simple **classification head** for binary classification:  
  `0 = neutral` vs `1 = hate / abusive`

## Features

## Installing Dependencies
    pip install -r requirements.txt

## Usage
### 1. Fine-tune the Model
Run:
```bash
python train.py
```
Saves logs, plots, and best model under `outputs`. 
<br>
<br>
If you want to disable SimCSE pretraining, set in `src/config.py`:
```bash
use_simcse: bool = False
```
### 2. Run the Streamlit App


