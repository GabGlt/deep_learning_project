# Final Project for Deep Learning Course

This repository contains the code for a deep learning project on **detecting hate and abusive language in Indonesian tweets** using:

- **IndoBERT** as the base language model
- **Unsupervised SimCSE** for contrastive pretraining of sentence embeddings
- A simple **classification head** for binary classification:  
  `0 = neutral` vs `1 = hate / abusive`

## Features

The application provides:

1. **Single text classification** – paste a tweet and get immediate prediction  
2. **Batch CSV prediction** – upload a CSV file with multiple texts and get predictions for all  
3. **Prediction confidence** – see probability scores for each class  
4. **Clean and simple UI** – easy to interact with using Streamlit 

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
### 2. Run the Streamlit App Locally

To run the app on your local machine, follow these steps:

1. **Activate your Python environment**:

```bash
& "D:/Deep Learning/environments/deep_learning/Scripts/Activate.ps1"
```

2. **Navigate to the app folder**:

```bash
cd app
```

3. **Run the Streamlit App**:

```bash
streamlit run app.py
```

4. **Open the app in your browser**:

After running the above command, Streamlit will display a local URL in the terminal, usually:
```bash
http://localhost:8501
```
Open this URL in your browser to start interacting with the app.


### 3. Access the Deployed App
The app is also deployed online and can be accessed here:
https://indobert-hate-speech-classifier.streamlit.app/
