# Import packages
import streamlit as st
import altair as at
import plotly.express as px
import pandas as pd
import numpy as np
import joblib



# Load the model
pipeline_lr1 = joblib.load(open("../models/LGR.pkl", "rb"))
pipeline_lr2 = joblib.load(open("../models/SVM.pkl", "rb"))



# Functions
def predict_emotions(text):
    results = pipeline_lr1.predict([text])
    return results[0]

def predict_emotions1(text):
    results = pipeline_lr2.predict([text])
    return results[0]


def predict_probability(text):
    results = pipeline_lr1.predict_proba([text])
    return results

def predict_probability1(text):
    results = pipeline_lr2.predict_proba([text])
    return results


st.title("Know Your Emotions")
menu = ["Home", "About"]
choice = st.sidebar.selectbox("Menu", menu)
if choice=="Home":
    st.subheader("Home- Emotion in Text")
    with st.form(key="emotion_clf_form"):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label="Submit")

    if submit_text:
        col1, col2 = st.beta_columns(2)

        prediction1 = predict_emotions(raw_text)
        prediction2 = predict_emotions1(raw_text)
        probability1 = predict_probability(raw_text)
        probability2 = predict_probability1(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction 1")
            st.write(prediction1)
            st.write("Confidence: ", np.max(probability1))

            st.success("Prediction 2")
            st.write(prediction2)
            st.write("Confidence: ", np.max(probability2))

        with col2:
            st.success("Prediction Probability 1")
            prob_df = pd.DataFrame(probability1, columns=pipeline_lr1.classes_)
            prob_df_clean = prob_df.T.reset_index()
            prob_df_clean.columns = ['emotions','probability']
            fig = at.Chart(prob_df_clean).mark_bar().encode(x="emotions", y="probability", color="emotions")
            st.altair_chart(fig, use_container_width=True)
            
            st.success("Prediction Probability 2")
            prob_df = pd.DataFrame(probability2, columns=pipeline_lr2.classes_)
            prob_df_clean = prob_df.T.reset_index()
            prob_df_clean.columns = ['emotions','probability']
            fig = at.Chart(prob_df_clean).mark_bar().encode(x="emotions", y="probability", color="emotions")
            st.altair_chart(fig, use_container_width=True)
elif choice=="About":
    st.subheader("About")