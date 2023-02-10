import pandas as pd
import numpy as np
import streamlit as st
import whisper
import pytube
from pytube import YouTube
from streamlit_chat import message
import openai
from openai.embeddings_utils import get_embedding, distances_from_embeddings
import os
import pinecone
from dotenv import load_dotenv

# whisper
model = whisper.load_model('base')
output = ''
data = []
data_transcription = []
embeddings = []
mp4_video = ''
audio_file = ''

# Pinacone

# Uncomment this section if you want to save the embedding in pinecone
#load_dotenv()
# initialize connection to pinecone (get API key at app.pinecone.io)
# pinecone.init(
#     api_key=os.getenv("PINACONE_API_KEY"),
#     environment=os.getenv("PINACONE_ENVIRONMENT")
# )
array = []

# Uncomment this section if you want to upload your own video
# Sidebar
# with st.sidebar:
#     user_secret = st.text_input(label = ":blue[OpenAI API key]",
#                                 value="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",
#                                 placeholder = "Paste your openAI API key, sk-",
#                                 type = "password")
#     youtube_link = st.text_input(label = ":red[Youtube link]",
#                                 value="https://youtu.be/rQeXGvFAJDQ",
#                                 placeholder = "")
#     if youtube_link and user_secret:
#         youtube_video = YouTube(youtube_link)
#         video_id = pytube.extract.video_id(youtube_link)
#         streams = youtube_video.streams.filter(only_audio=True)
#         stream = streams.first()
#         if st.button("Start Analysis"):
#             if os.path.exists("word_embeddings.csv"):
#                 os.remove("word_embeddings.csv")
                
#             with st.spinner('Running process...'):
#                 # Get the video mp4
#                 mp4_video = stream.download(filename='youtube_video.mp4')
#                 audio_file = open(mp4_video, 'rb')
#                 st.write(youtube_video.title)
#                 st.video(youtube_link) 

#                 # Whisper
#                 output = model.transcribe("youtube_video.mp4")
                
#                 # Transcription
#                 transcription = {
#                     "title": youtube_video.title.strip(),
#                     "transcription": output['text']
#                 }
#                 data_transcription.append(transcription)
#                 pd.DataFrame(data_transcription).to_csv('transcription.csv') 
#                 segments = output['segments']

#                 # Pinacone index
#                 # check if index_name index already exists (only create index if not)
#                 # index_name = str(video_id)
#                 # # check if 'index_name' index already exists (only create index if not)
#                 # if 'index1' not in pinecone.list_indexes():
#                 #     pinecone.create_index('index1', dimension=len(segments))
#                 # # connect to index
#                 # index = pinecone.Index('index1')
                
#                 #st.write(segments)
#                 #Embeddings
#                 for segment in segments:
#                     openai.api_key = user_secret
#                     response = openai.Embedding.create(
#                         input= segment["text"].strip(),
#                         model="text-embedding-ada-002"
#                     )
#                     embeddings = response['data'][0]['embedding']
#                     meta = {
#                         "text": segment["text"].strip(),
#                         "start": segment['start'],
#                         "end": segment['end'],
#                         "embedding": embeddings
#                     }
#                     data.append(meta)
#                 # upsert_response = index.upsert(
#                 #         vectors=data,
#                 #         namespace=video_id
#                 #     )
#                 pd.DataFrame(data).to_csv('word_embeddings.csv') 
#                 os.remove("youtube_video.mp4")
#                 st.success('Analysis completed')

st.markdown('<h1>Youtube GPT 🤖<small> by Code GPT</small></h1>', unsafe_allow_html=True)
st.write("Start a chat with this video of Microsoft CEO Satya Nadella's interview. You just need to add your OpenAI API Key and paste it in the 'Chat with the video' tab.")

DEFAULT_WIDTH = 80
VIDEO_DATA = "https://youtu.be/bsFXgfbj8Bc"

width = 40

width = max(width, 0.01)
side = max((100 - width) / 2, 0.01)

_, container, _ = st.columns([side, 47, side])
container.video(data=VIDEO_DATA)
tab1, tab2, tab3, tab4 = st.tabs(["Intro", "Transcription", "Embedding", "Chat with the Video"])
with tab1:
    st.markdown('Read the article to know how it works: [Medium Article]("https://medium.com/@dan.avila7/youtube-gpt-start-a-chat-with-a-video-efe92a499e60")')
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
    st.write("This software was developed by Code GPT, for more information visit: https://codegpt.co")
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
    user_secret = st.text_input(label = ":blue[OpenAI API key]",
                                placeholder = "Paste your openAI API key, sk-",
                                type = "password")
    st.write('To obtain an API Key you must create an OpenAI account at the following link: https://openai.com/api/')
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    def get_text():
        if user_secret:
            st.header("Ask me something about the video:")
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
        for i, row in df.sort_values('distances', ascending=True).head(4).iterrows():
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
