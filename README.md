<h1 align="center">
YoutubeGPT 🤖
</h1>
With Youtube GPT you will be able to extract all the information from a video on YouTube just by pasting the video link.
You will obtain the transcription, the embedding of each segment and also ask questions to the video through a chat.

All code was written with the help of <a href="https://codegpt.co">Code GPT</a>
<hr>
<br>

# 🔧 Features

- Video transcription with **OpenAI Whisper**
- Embedding Transcript Segments with the OpenAI API (**text-embedding-ada-002**)
- Chat with the video using **streamlit-chat** and OpenAI API (**text-davinci-003**)

# Dependencies

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

To install the dependencies with this file, you will need to use the pip command. The pip command will read the requirements.txt file and download and install all the dependencies listed in it. The command is the following:

```bash
pip install -r requirements.txt
```