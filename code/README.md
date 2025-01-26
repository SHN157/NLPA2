# NLP Assignment 2 (AIT - DSAI)

- [Student Information](#student-information)
- [Files Structure](#files-structure)
- [Dataset Description](#dataset-description)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Hyperparameters](#hyperparameters)
- [Web Application Documentation](#web-application-documentation)
- [Result](#result)



## Student Information
- Name: Soe Htet Naing
- ID: st125166



## Files Structure
Contains 2 folders named app and code

'code' folder contains
- `LSTM_LM.ipynb`: Notebook containing the training process of the LSTM language model.

'app' folder contains models folder and
- `app.py`: Entry point for the web application, built using Streamlit

'models' folder contains
- `utils.py`: Contains utility functions such as the text generation logic.
- `classes.py`: Includes the `LSTMLanguageModel` class definition.
- `args.pkl`: Metadata file containing model parameters and vocabulary.
- `best-val-lstm_lm.pt`: Trained LSTM model state dictionary.



## Dataset Description
- **Data Source**: The dataset used for this project is the text of *Anna Karenina* by Leo Tolstoy.
- **Hugging Face Integration**: A Hugging Face dataset repository was created for storage and management.
- **Dataset Statistics**:
  - Training set: 32,180 rows
  - Validation set: 4,022 rows
  - Test set: 4,023 rows
- The dataset was split into training, validation, and testing sets to facilitate model development and evaluation.



## Data Preprocessing
1. **Tokenizing**:
   - Text is tokenized using the `basic_english` tokenizer from `torchtext`.
   - Tokens are stored in a `tokens` field.
2. **Numericalizing**:
   - Vocabulary is built using `torchtext.vocab.build_vocab_from_iterator`.
   - Tokens with a minimum frequency of 3 are included.
   - Special tokens like `<unk>` and `<eos>` are added.
   - The vocabulary maps words to indices and vice versa.


## Prepare Batch Loader
The `get_data` function prepares data for the language model in batch format:

- **Steps**:
  1. Tokens from each example are processed and appended with the `<eos>` token.
  2. Each token is mapped to its corresponding index using the vocabulary.
  3. The tokenized data is combined into a single tensor and truncated to fit full batches.
  4. Data is reshaped into a [batch_size, sequence_length] format.

  The function is used to create batches with size = 128 for training, validation, and testing datasets:



## Model Architecture

The LSTM Language Model is designed to generate text based on sequential input. Its key components are:

1. **Embedding Layer**: Converts input word indices into dense vector representations of size `emb_dim`.
2. **LSTM Layer**: Processes the embeddings sequentially, producing hidden states and outputs. Configured with `num_layers`, `hid_dim`, and dropout for better generalization.
3. **Dropout Layer**: Adds regularization by randomly dropping connections during training.
4. **Linear Layer**: Maps LSTM outputs to vocabulary scores for the next word prediction.
5. **Initialization**: Embedding, linear, and LSTM weights are initialized uniformly for better convergence.
6. **Hidden State Management**:
   - `init_hidden`: Creates initial hidden and cell states for the LSTM.
   - `detach_hidden`: Detaches hidden states from the computational graph to prevent gradient tracking during backpropagation.

The model takes a batch of input sequences, embeds them, processes them through LSTM, applies dropout, and outputs predictions for each word in the sequence.



## Training Process
- Optimizer: Adam with a learning rate.
- Loss Function: CrossEntropyLoss.
- Hidden states are initialized for the LSTM during each batch.



## Hyperparameters
- `vocab_size`: Vocabulary size 6013.
- `emb_dim`: 1024 Embedding dimension.
- `hid_dim`: 1024 LSTM hidden state dimension.
- `num_layers`: 2 Number of LSTM layers.
- `dropout_rate`: Dropout probability 0.65.
- `lr`: Learning rate 1e-3.



## Web Application Documentation

### How to run
1. Ensure all dependencies are installed.
2. Navigate to the `app` folder.
3. Run `streamlit run app.py`.
4. Open `http://localhost:8501/` in your browser.

Screenshots of testing web are shown in samples_images.

### Usage
- Enter a text prompt in the input box.
- Select the desired creativity level (temperature) and sequence length.
- Click "Generate Text" to see the model's output.

### Documentation
1. **Model Loading**:
   - The trained model (`best-val-lstm_lm.pt`) and metadata (`args.pkl`) are loaded at runtime.
2. **Text Generation**:
   - The `generate` function from `utils.py` handles text generation using the LSTM LanguageModel from `classes.py`.
   - Tokens are predicted sequentially, with unknown and end-of-sequence tokens handled appropriately.
3. **Streamlit Integration**:
   - User input is taken via Streamlit widgets.
   - Generated text is displayed dynamically based on model predictions.



## Result
- The model's performance is measured using perplexity on train,test,valid datasets , with lower values indicating better results:
  - Train Perplexity: 54.913
  - Validation Perplexity: 73.194
  - Test Perplexity: 73.936
