import streamlit as st
import pickle
import torch
from models.utils import generate
from models.classes import LSTMLanguageModel

# Load model configuration and tokenizer
Data = pickle.load(open('./models/args.pkl', 'rb'))
vocab_size = Data['vocab_size']
emb_dim = Data['emb_dim']
hid_dim = Data['hid_dim']
num_layers = Data['num_layers']
dropout_rate = Data['dropout_rate']
tokenizer = Data['tokenizer']
vocab = Data['vocab']

# Load model
model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate)
model.load_state_dict(torch.load('./models/best-val-lstm_lm.pt', map_location=torch.device('cpu')))
model.eval()

# Streamlit app setup
st.title("LSTM Language Model Text Generator")
st.write("Generate text continuations based on your input prompt!")

# User inputs
prompt = st.text_input("Enter a text prompt:", "")
temperature = st.selectbox(
    "Select creativity level (temperature):",
    options=[0.5, 0.7, 0.75, 0.8, 1.0],
    index=3  # Default to 0.8
)
seq_len = st.selectbox(
    "Select sequence length:",
    options=[5, 10, 15, 20, 25, 30],
    index=5  # Default to 30
)

# Fixed seed
seed = 0

# Generate text when the button is clicked
if st.button("Generate Text"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with st.spinner("Generating..."):
        generated_tokens = generate(prompt, seq_len, temperature, model, tokenizer, vocab, device, seed)
    generated_text = " ".join(generated_tokens)
    st.write("### Generated Text:")
    st.write(generated_text)

# Footer
st.write("Choose the sequence length and creativity to experiment with the generated output.")
