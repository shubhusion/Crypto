import pickle
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.graph_objects as go

class CipherPredictor:
    """Class to predict the type of columnar cipher using a trained LSTM model."""

    def __init__(
        self, model_path="models/lstm_model/lstm_model.h5", tokenizer_path="models/lstm_model/tokenizer.pkl"
    ):
        """Initialize the predictor with model and tokenizer."""
        # Load the trained model
        self.model = load_model(model_path)

        # Load the tokenizer
        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        self.max_length = 100  # Same as training

    def predict(self, text):
        """Predict the probability of the given text being a complete columnar cipher."""
        # Preprocess the input text
        text_seq = self.tokenizer.texts_to_sequences([text])
        text_pad = pad_sequences(text_seq, maxlen=self.max_length)

        # Make prediction
        prediction = self.model.predict(text_pad)
        probability = prediction[0][0]

        return probability


def main():
    """Main function to run the Streamlit app for columnar cipher classification."""
    st.set_page_config(
        page_title="Columnar Cipher Classifier", page_icon="🔒", layout="wide"
    )

    st.title("🔐 Columnar Cipher Type Classifier")
    st.markdown(
        """
    This application classifies columnar ciphertexts into two categories:
    - Complete Columnar Cipher
    - Incomplete Columnar Cipher
    """
    )

    # Initialize predictor
    try:
        predictor = CipherPredictor()
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        model_loaded = False

    # Input section
    st.header("Input Ciphertext")
    input_method = st.radio("Choose input method:", ("Text Input", "File Upload"))

    if input_method == "Text Input":
        cipher_text = st.text_area("Enter your ciphertext here:", height=150)
    else:
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        if uploaded_file:
            cipher_text = uploaded_file.read().decode()
            st.text_area("File contents:", cipher_text, height=150)
        else:
            cipher_text = ""

    # Predict button
    if st.button("Classify Cipher") and model_loaded:
        if cipher_text:
            try:
                # Make prediction
                probability = predictor.predict(cipher_text)

                # Display results
                st.header("Classification Results")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Prediction", "Complete" if probability > 0.5 else "Incomplete"
                    )

                with col2:
                    st.metric(
                        "Confidence",
                        f"{abs(probability if probability > 0.5 else 1-probability)*100:.2f}%",
                    )

                # Visualization
                st.header("Probability Distribution")

                fig = go.Figure(
                    go.Bar(
                        x=["Incomplete", "Complete"],
                        y=[1 - probability, probability],
                        text=[f"{(1-probability)*100:.1f}%", f"{probability*100:.1f}%"],
                        textposition="auto",
                    )
                )

                fig.update_layout(
                    title="Prediction Probabilities",
                    yaxis_title="Probability",
                    yaxis_range=[0, 1],
                )

                st.plotly_chart(fig)

                # Additional Statistics
                st.header("Text Statistics")
                stats_col1, stats_col2 = st.columns(2)

                with stats_col1:
                    st.metric("Text Length", len(cipher_text))
                    st.metric("Unique Characters", len(set(cipher_text)))

                with stats_col2:
                    char_freq = pd.Series(list(cipher_text)).value_counts()
                    st.metric("Most Common Character", char_freq.index[0])
                    st.metric("Least Common Character", char_freq.index[-1])

                # Character frequency visualization
                st.subheader("Character Frequency Distribution")
                char_freq_df = pd.DataFrame(
                    {"Character": char_freq.index, "Frequency": char_freq.values}
                )

                fig = go.Figure(
                    go.Bar(
                        x=char_freq_df["Character"][:10],  # Top 10 characters
                        y=char_freq_df["Frequency"][:10],
                        text=char_freq_df["Frequency"][:10],
                        textposition="auto",
                    )
                )

                fig.update_layout(
                    title="Top 10 Most Frequent Characters",
                    xaxis_title="Character",
                    yaxis_title="Frequency",
                )

                st.plotly_chart(fig)

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            st.warning("Please enter or upload some text first.")

    # About section
    st.sidebar.header("About")
    st.sidebar.markdown(
        """
    This application uses a deep learning model (LSTM) to classify columnar ciphers.
    
    **Complete Columnar Cipher:**
    - Uses padding to fill incomplete blocks
    - Regular pattern structure
    
    **Incomplete Columnar Cipher:**
    - No padding in incomplete blocks
    - May have irregular endings
    """
    )

    # App information
    st.sidebar.header("App Information")
    st.sidebar.info(
        """
    Version: 1.0.0  
    Last Updated: November 2024  
    Author: Your Name
    """
    )


if __name__ == "__main__":
    main()
