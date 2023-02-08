import pandas as pd
import numpy as np
import streamlit as st
import whisper
from pytube import YouTube
from streamlit_chat import message
import openai
from openai.embeddings_utils import get_embedding, distances_from_embeddings
import os

# whisper
model = whisper.load_model('base')
output = ''
data = []
embeddings = []

# Sidebar
with st.sidebar:
    user_secret = st.text_input(label = ":blue[OpenAI API key]",
                                placeholder = "Paste your openAI API key, sk-",
                                type = "password")
    youtube_link = st.text_input(label = ":red[Youtube link]",
                                placeholder = "")
    if youtube_link and user_secret:
        youtube_video = YouTube(youtube_link)
        streams = youtube_video.streams.filter(only_audio=True)
        stream = streams.first()
        if st.button("Start Analysis"):
            if os.path.exists("word_embeddings.csv"):
                os.remove("word_embeddings.csv")
                
            with st.spinner('Running process...'):
                mp4_video = stream.download(filename='youtube_video.mp4')
                audio_file = open(mp4_video, 'rb')
                st.write(youtube_video.title)
                #audio_bytes = audio_file.read()
                #st.audio(audio_bytes, format='audio/ogg')
                st.video(youtube_link) 
                output = model.transcribe("youtube_video.mp4")
                segments = output['segments']
                for segment in segments:
                    openai.api_key = user_secret
                    response = openai.Embedding.create(
                        input= segment["text"].strip(),
                        model="text-embedding-ada-002"
                    )
                    embeddings = response['data'][0]['embedding']
                    meta = {
                        "text": segment["text"].strip(),
                        "start": segment['start'],
                        "end": segment['end'],
                        "embedding": embeddings
                    }
                    data.append(meta)
                pd.DataFrame(data).to_csv('word_embeddings.csv') 
                st.success('Analysis completed')

st.title("Youtube GPT 🤖 ")
tab1, tab2 = st.tabs(["Transcription", "Chat with the Video"])
with tab1: 
    st.header("OpenAI Whisper:")
    if os.path.exists("word_embeddings.csv"):
        df = pd.read_csv('word_embeddings.csv')
        st.write(df)
with tab2:
    st.header("Ask me something about the video:")
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    def get_text():
        input_text = st.text_input("You: ","", key="input")
        return input_text

    user_input = get_text()

    def get_embedding_text(api_key, prompt):
        openai.api_key = user_secret
        response = openai.Embedding.create(
            input= prompt.strip(),
            model="text-embedding-ada-002"
        )
        q_embedding = response['data'][0]['embedding']
        df=pd.read_csv('word_embeddings.csv', index_col=0)
        df['embedding'] = df['embedding'].apply(eval).apply(np.array)

        df['distances'] = distances_from_embeddings(q_embedding, df['embedding'].values, distance_metric='cosine')
        df.sort_values("distances", ascending=False).head(1)
        return df['text'][0]

    def generate_response(api_key, prompt):
        one_shot_prompt = '''I am YoutubeGPT, a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer.
        Q: What is human life expectancy in the United States?
        A: Human life expectancy in the United States is 78 years.
        Q: '''+prompt+'''
        A: '''
        completions = openai.Completion.create(
            engine = "text-davinci-003",
            prompt = one_shot_prompt,
            max_tokens = 1024,
            n = 1,
            stop=["Q:"],
            temperature=0.5,
        )
        message = completions.choices[0].text
        return message

    if user_input:
        text_embedding = get_embedding_text(user_secret, user_input)
        user_input_embedding = 'with this context "'+text_embedding+'", answer this question'+user_input
        output = generate_response(user_secret, user_input_embedding)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')





