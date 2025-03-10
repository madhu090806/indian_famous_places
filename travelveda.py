import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="TravelVeda", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("TravelVeda")
st.write("Welcome to TravelVeda! Upload a JSON file containing information about famous places in India and ask your questions.")

# File uploader for JSON
uploaded_file = st.file_uploader("Upload a JSON file", type="json")

if uploaded_file:
    try:
        data = json.load(uploaded_file)
        qa_list = []
        
        # Convert JSON to structured Q&A format
        for state, places in data.items():
            for place in places:
                qa_list.append({"Question": f"Give me the information related to {place.get('name', 'Unknown Place')}.", "Answer": place.get("info", "No information available.")})
                qa_list.append({"Question": f"Where is {place.get('name', 'Unknown Place')} located?", "Answer": f"{place.get('name', 'Unknown Place')} is located in {state}, India."})
                qa_list.append({"Question": f"What is the historical significance of {place.get('name', 'Unknown Place')}?", "Answer": place.get("info", "No historical information available.")})
                qa_list.append({"Question": f"What is the architectural style of {place.get('name', 'Unknown Place')}?", "Answer": place.get("architecture", "No architectural details available.")})
                qa_list.append({"Question": f"How can I visit {place.get('name', 'Unknown Place')}?", "Answer": place.get("visiting_guide", "No visiting guide available.")})
        
        df = pd.DataFrame(qa_list)
        df = df.fillna("")
        df['Question'] = df['Question'].str.lower()
        df['Answer'] = df['Answer'].str.lower()

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer()
        question_vectors = vectorizer.fit_transform(df['Question'])

        # Configure Generative AI
        API_KEY = "AIzaSyC-VNtSTpG6TAynZRgmAuAySWj7iD0q_jc"
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')

        def find_closest_question(user_query, vectorizer, question_vectors, df):
            query_vector = vectorizer.transform([user_query.lower()])
            similarities = cosine_similarity(query_vector, question_vectors).flatten()
            best_match_index = similarities.argmax()
            best_match_score = similarities[best_match_index]
            if best_match_score > 0.3:  
                return df.iloc[best_match_index]['Answer']
            else:
                return None

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Type your question here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            closest_answer = find_closest_question(prompt, vectorizer, question_vectors, df)
            if closest_answer:
                st.session_state.messages.append({"role": "assistant", "content": closest_answer})
                with st.chat_message("assistant"):
                    st.markdown(closest_answer)
            else:
                try:
                    response = model.generate_content(prompt)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    with st.chat_message("assistant"):
                        st.markdown(response.text)
                except Exception as e:
                    st.error(f"Sorry, I couldn't generate a response. Error: {e}")

    except Exception as e:
        st.error(f"Error processing JSON file: {e}")
