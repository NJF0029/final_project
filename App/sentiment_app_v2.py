import streamlit as st
import pandas as pd
import joblib,os

# NLP Packages
import spacy
nlp = spacy.load("en_core_web_sm")
from nltk.corpus import wordnet
import string
import re
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup as bs
nltk.download("stopwords")
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# EDA Packages
import pandas as pd

# worldcloud
from wordcloud import WordCloud
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Vectoriser
#review_sentiment_vectoriser = open("<vectorizer.pckl>","rb")
#review_cv = joblib.load(review_sentiment_vectoriser)

#Load  Models
def load_classification_model(model_file):
    loaded_models = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_models

def get_keys(val,my_dict):
    for key,value in my_dict.items():
        if val == value:
            return key

def get_wordnet_pos(pos_tag):

    '''
    Function to return the corresponding wordnet object value of the
    'Part Of Speach tag' i.e "thing : NN" corresponds to Noun in wordnet.
    Input: string (pos tag)
    Output: string (a wordnet object)
    '''

    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def review_cleaner(review):

    '''
    Function to convert a review to a tokenised string of words.
    Input: string (a raw movie review)
    Output: string (a preprocessed movie review)
    '''

    # 1. Remove HTML
    review = bs(review).get_text()

    # 2. Remove non-letters
    review = re.sub("[^a-zA-Z]", " ", review)

    # 3. split by just words
    review = re.split(r'\W+',review)
    review = " ".join(review)

    # 4. Convert to lower case
    review = review.lower()

    # 5. split into individual words
    review = review.split()

    # 6. Import stop words
    stop = (stopwords.words("english"))

    # 7. Remove stop words
    review = [t for t in review if t not in stop]

    # 8. remove empty tokens
    review = [t for t in review if len(t) > 0]

    # 9. pos tag text
    pos_tags = pos_tag(review)

    # 10. lemmatize text
    review = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]

    # 12. remove words with only one letter
    review = [t for t in review if len(t) > 1]

    # 13. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( review ))

def main():
    """ Movie Sentiment Classifier """
    st.title("Movie Sentiment Classifier App")
    st.subheader("Nathan Fournillier DAFT OCT 2020")

    activities = ["Classification", "NLP"]

    choice = st.sidebar.selectbox("Choose Activity", activities)

    if choice == 'Classification':
        st.info("Classification with ML")

        movie_review  = st.text_area("Enter Your Text", "Type Here")
        ml_models_list = ["RForest"]
        model_choice = st.selectbox("Choose ML Model", ml_models_list)
        prediction_labels = {'Negative':0, 'Positive':1}
        if st.button("Classify"):
            st.text(f"Original text ::\n{movie_review}")
            vect_text = review_cv.transform([movie_review]).toarray()
            if model_choice == 'LogReg':
                predictor = load_classification_model("<model_file_name>")
                prediction = predictor.predict(vect_text)
                st.write(prediction)
                final_result = get_keys(prediction,prediction_labels)
                st.success("Review Categorised as:: {}".format(final_reslut))

    if choice == 'NLP':
        st.info("Natural Language Processing")
        movie_review  = st.text_area("Enter Your Text", "Type Here")
        nlp_task = ["Tokenisation","Lemmatisation","NER","POS_Tags","Clean All"]
        task_choice = st.selectbox("Choose NLP Task", nlp_task)
        if st.button("Analyse"):
            st.info("Original Text: {}".format(movie_review))

            docx = nlp(movie_review)
            if task_choice == 'Tokenisation':
                result = [token.text for token in docx]
                st.json(result)
            elif task_choice == 'Lemmatisation':
                result = ["'Token':{}, 'Lemma':{}".format(token.text,token.lemma_) for token in docx]
                st.json(result)
            elif task_choice == 'NER':
                result = [(entity.text,entity.label_)for entity in docx.ents]
                st.json(result)
            elif task_choice == 'POS_Tags':
                result = ["'Token':{}, 'POS':{}, 'Dependency':{}".format(word.text,word.tag_,word.dep_) for word in docx]
            elif task_choice == 'Clean All':
                result = [review_cleaner(movie_review)]
                st.json(result)

        if st.button("Wordcloud"):
            wordcloud = WordCloud().generate(movie_review)
            plt.imshow(wordcloud,interpolation='bilinear')
            plt.axis("off")
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)

if __name__ == '__main__':
    main()
