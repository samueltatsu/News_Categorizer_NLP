# Import Essential Library
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load Model
model = load_model('model.H5')

# Function to run model predictor
def run():
    # Set Title
    st.title('News Categorizer')

    st.markdown('---')

    # Insert Image
    st.image('https://www.thetimes.co.uk/d/img/logos/times-black-ee1e0ce4ed.png')

    # Creating Form for Data Inference
    st.markdown('## Input Data')
    with st.form('my_form'):
        headline = st.text_input('Headline', 'Enter News Headline'),
        summary = st.text_input('Summary', 'Enter News Summary')

        submitted = st.form_submit_button("Check")

    # Dataframe
    data = {
        'headline': headline,
        'summary': summary,
    }
    df = pd.DataFrame(data)

    # display dataframe of inputted data
    st.dataframe(df)

    # concat headline and summary
    text_df = pd.DataFrame()
    text_df['text'] = df['headline'] + ' ' + df['summary']

    ## Preprocessing
    # initialize necessary packages
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # set stopwords and lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # define function to preprocess text
    def preprocess_text(text):
        # Case folding
        words = text.lower()
        # Mention removal
        words = re.sub("@[A-Za-z0-9_]+", " ", text)
        # Hashtags removal
        words = re.sub("#[A-Za-z0-9_]+", " ", text)
        # Newline removal (\n)
        words = re.sub(r"\\n", " ",text)
        # Whitespace removal
        words = text.strip()
        # URL removal
        words = re.sub(r"http\S+", " ", text)
        words = re.sub(r"www.\S+", " ", text)
        # Non-letter removal (such as emoticon, symbol (like μ, $, 兀), etc
        words = re.sub("[^A-Za-z\s']", " ", text)

        # tokenize text
        tokens = word_tokenize(text)
        # remove stopwords
        tokens = [word for word in tokens if word not in stop_words]
        # lemmatize words
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        # join the words back into a single string
        return ' '.join(tokens)
    
    # apply text preprocessing
    text_df['text'] = text_df['text'].apply(preprocess_text)

    # show result
    if submitted:
        # Perform prediction
        predictions = model.predict(text_df)

        # Get the index of the class with the highest probability for each prediction
        predicted_class = np.argmax(predictions)

        categories_dict = {0:"WELLNESS", 1:"POLITICS", 2:"ENTERTAINMENT", 3:"TRAVEL", 4:"STYLE & BEAUTY", 5:"PARENTING", 6:"FOOD & DRINK", 7:"WORLD NEWS", 8:"BUSINESS", 9:"SPORTS"}

        # Get the predicted category
        result = categories_dict[predicted_class]
        
        st.write(f"The news falls into category: {result}")

if __name__=='__main__':
    run()