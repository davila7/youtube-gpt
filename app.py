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
data_transcription = []
embeddings = []
mp4_video = ''
audio_file = ''
# Sidebar
with st.sidebar:
    user_secret = st.text_input(label = ":blue[OpenAI API key]", value = "",
                                placeholder = "Paste your openAI API key, sk-",
                                type = "password")
    youtube_link = st.text_input(label = ":red[Youtube link]", value = "",
                                placeholder = "")
    if youtube_link and user_secret:
        youtube_video = YouTube(youtube_link)
        streams = youtube_video.streams.filter(only_audio=True)
        stream = streams.first()
        if st.button("Start Analysis"):
            if os.path.exists("word_embeddings.csv"):
                os.remove("word_embeddings.csv")
                
            with st.spinner('Running process...'):
                # Get the video mp4
                mp4_video = stream.download(filename='youtube_video.mp4')
                audio_file = open(mp4_video, 'rb')
                st.write(youtube_video.title)
                st.video(youtube_link) 

                # Whisper
                output = model.transcribe("youtube_video.mp4")
                
                # Transcription
                transcription = {
                    "title": youtube_video.title.strip(),
                    "transcription": output['text']
                }
                data_transcription.append(transcription)
                pd.DataFrame(data_transcription).to_csv('transcription.csv') 

                # Embeddings
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
tab1, tab2, tab3, tab4 = st.tabs(["Intro", "Transcription", "Embedding", "Chat with the Video"])
with tab1:
    st.write("To start, add your OpenAI Api key and the url of the video on Youtube. Then click Start Analysis")
    st.markdown("### How does it work?")
    st.write("Youtube GPT was written with the following tools:")
    st.markdown("#### Code GPT")
    st.write("All code was written with the help of Code GPT. Visit [codegpt.co]('https://codegpt.co') to get the extension.")
    st.markdown("#### Streamlit")
    st.write("The design was written with [Streamlit]('https://streamlit.io/').")
    st.markdown("#### Whisper")
    st.write("Video transcription is done by [OpenAI Whisper]('https://openai.com/blog/whisper/').")
    st.markdown("#### Embedding")
    st.write('[Embedding]("https://platform.openai.com/docs/guides/embeddings") is done via the OpenAI API with "text-embedding-ada-002"')
    st.markdown("#### GPT-3")
    st.write('The chat uses the OpenAI API with the [GPT-3]("https://platform.openai.com/docs/models/gpt-3") model "text-davinci-003""')
    st.markdown("""---""")
    st.write('Repo: [Github](https://github.com/davila7/youtube-gpt)')
    st.write('Article : [Medium](https://)')

with tab2: 
    st.header("Transcription:")
    if(os.path.exists("youtube_video.mp4")):
        audio_file = open('youtube_video.mp4', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/ogg')
    if os.path.exists("transcription.csv"):
        df = pd.read_csv('transcription.csv')
        st.write(df)
with tab3:
    st.header("Embedding:")
    if os.path.exists("word_embeddings.csv"):
        df = pd.read_csv('word_embeddings.csv')
        st.write(df)
with tab4:
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
        returns = []
        
        # Sort by distance with 2 hints
        for i, row in df.sort_values('distances', ascending=True).head(2).iterrows():
            # Else add it to the text that is being returned
            returns.append(row["text"])

        # Return the context
        return "\n\n###\n\n".join(returns)

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
        title = pd.read_csv('transcription.csv')['title']
        string_title = "\n\n###\n\n".join(title)
        user_input_embedding = 'Using this context: "'+string_title+'. '+text_embedding+'", answer the following question. \n'+user_input
        # st.write(user_input_embedding)
        output = generate_response(user_secret, user_input_embedding)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')





