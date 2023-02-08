import pandas as pd
import numpy as np
import streamlit as st
import whisper
from pytube import YouTube
from streamlit_chat import message
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
#openai.api_key = st.secrets["apikey"]

# whisper
model = whisper.load_model('base')
output = ''

# Sidebar
with st.sidebar:
    user_secret = st.text_input(label = ":blue[OpenAI API key]",
                                placeholder = "Paste your openAI API key, sk-",
                                type = "password")
    youtube_link = st.text_input(label = ":red[Youtube link]",
                                placeholder = "")
    if youtube_link:
        youtube_video = YouTube(youtube_link)
        streams = youtube_video.streams.filter(only_audio=True)
        stream = streams.first()
        if st.button("Start Analysis"):
            with st.spinner('Running process...'):
                mp4_video = stream.download(filename='youtube_video.mp4')
                audio_file = open(mp4_video, 'rb')
                st.write(youtube_video.title)
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/ogg')
                output = model.transcribe("youtube_video.mp4")
                st.success('Analysis completed')

st.title("Youtube GPT 🤖 ")
# chart_data = pd.DataFrame(np.random.randn(20, 3),columns=['a', 'b', 'c'])
# with tab2:
#     st.area_chart(chart_data)
tab1, tab2 = st.tabs(["Transcription", "Chat with the Video"])
with tab1: 
    st.header("OpenAI Whisper:")
    st.write(output)
with tab2:
    # Storing the chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    def get_text():
        input_text = st.text_input("You: ","", key="input")
        return input_text

    user_input = get_text()

    def generate_response(api_key, prompt):
        openai.api_key = api_key
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
        output = generate_response(user_secret, user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')





