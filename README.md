<h1 align="center">
YoutubeGPT 🤖
</h1>

Read the article to know how it works: <a href="https://medium.com/@dan.avila7/youtube-gpt-start-a-chat-with-a-video-efe92a499e60">Medium Article</a>

With Youtube GPT you will be able to extract all the information from a video on YouTube just by pasting the video link.
You will obtain the transcription, the embedding of each segment and also ask questions to the video through a chat.

All code was written with the help of <a href="https://codegpt.co">Code GPT</a>

<a href="https://codegpt.co" target="_blank"><img width="753" alt="Captura de Pantalla 2023-02-08 a la(s) 9 16 43 p  m" src="https://user-images.githubusercontent.com/6216945/217699939-eca3ae47-c488-44da-9cf6-c7caef69e1a7.png"></a>

<hr>
<br>

# Features

- Video transcription with **OpenAI Whisper**
- Embedding Transcript Segments with the OpenAI API (**text-embedding-ada-002**)
- Chat with the video using **streamlit-chat** and OpenAI API (**text-davinci-003**)

# Example
For this example we are going to use this video from The PyCoach
https://youtu.be/lKO3qDLCAnk

Add the video URL and then click Start Analysis
![Youtube](https://user-images.githubusercontent.com/6216945/217701635-7c386ca7-c802-4f56-8148-dcce57555b5a.gif)

## Pytube and OpenAI Whisper
The video will be downloaded with pytube and then OpenAI Whisper will take care of transcribing and segmenting the video.
![Pyyube Whisper](https://user-images.githubusercontent.com/6216945/217704219-886d0afc-4181-4797-8827-82f4fd456f4f.gif)

```python
# Get the video 
youtube_video = YouTube(youtube_link)
streams = youtube_video.streams.filter(only_audio=True)
mp4_video = stream.download(filename='youtube_video.mp4')
audio_file = open(mp4_video, 'rb')

# whisper load base model
model = whisper.load_model('base')

# Whisper transcription
output = model.transcribe("youtube_video.mp4")
```

## Embedding with "text-embedding-ada-002"
We obtain the vectors with **text-embedding-ada-002** of each segment delivered by whisper
![Embedding](https://user-images.githubusercontent.com/6216945/217705008-180285d7-6bce-40c3-8601-576cc2f38171.gif)

```python
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
```
## OpenAI GPT-3
We make a question to the vectorized text, we do the search of the context and then we send the prompt with the context to the model "text-davinci-003"

![Question1](https://user-images.githubusercontent.com/6216945/217708086-b89dce2e-e3e2-47a7-b7dd-77e402d818cb.gif)

We can even ask direct questions about what happened in the video. For example, here we ask about how long the exercise with Numpy that Pycoach did in the video took.

![Question2](https://user-images.githubusercontent.com/6216945/217708485-df1edef3-d5f1-4b4a-a5c9-d08f31c80be4.gif)

# Running Locally

1. Clone the repository

```bash
git clone https://github.com/davila7/youtube-gpt
cd youtube-gpt
```
2. Install dependencies

These dependencies are required to install with the requirements.txt file:

* streamlit 
* streamlit_chat 
* matplotlib 
* plotly 
* scipy 
* sklearn 
* pandas 
* numpy 
* git+https://github.com/openai/whisper.git 
* pytube 
* openai-whisper

```bash
pip install -r requirements.txt
```
3. Run the Streamlit server

```bash
streamlit run app.py
```

## Upcoming Features 🚀

- Semantic search with embedding
- Chart with emotional analysis
- Connect with Pinecone
