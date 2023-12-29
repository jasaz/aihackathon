# Overview

This piece of code was writtten for Hackathon with the theme #aihackathon. The idea was to assist the scrum master using AI. It consisted of two parts:
- Provide summary and sentiment analysis of the meeting based on audio conversation during the scrum meeting
- Predict the storypoint of the task

  ## Summary and Sentiment analysis
- *summarize_sentiment.py* : This code reads the audio file from the GCS bucket and converts it into audio transcript using Google's Speech to Text API. The transcript is then used to generate the summary and sentiment using OpenAI. 
