from google.cloud import speech, speech_v1p1beta1
import time
import openai

openai.api_key = 'abc'

# returns audio transcript
def return_transcript(gcs_uri): 

    # Create a client
    client = speech_v1p1beta1.SpeechClient()

    # Read from GCS bucket
    # The audio needs to be uploaded to the GCS bucket

    #gs_uri="gs://cloud-ai-platform-hackathon23-samurais-sample-audio-mumbai/SamurAIs-Retro-Meeting-1.mp3"
    audio = speech_v1p1beta1.RecognitionAudio(uri= gcs_uri)

    config = speech_v1p1beta1.RecognitionConfig(
            encoding=speech_v1p1beta1.RecognitionConfig.AudioEncoding.MP3,
            sample_rate_hertz=44100,
            #enable_automatic_punctuation=True,
            language_code="en-US",
        )

    # Detects speech in the audio file
    operation = client.long_running_recognize(config=config, audio=audio)

    while not operation.done():
        print('Waiting for operation to complete...')
        time.sleep(10)

    response = operation.result()

    # Prepare the transcript

    transcript_builder = []
        # Each result is for a consecutive portion of the audio. Iterate through
        # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        transcript_builder.append(f"\nTranscript: {result.alternatives[0].transcript}")
        transcript_builder.append(f"\nConfidence: {result.alternatives[0].confidence}")

    transcript = "".join(transcript_builder)
    
    return transcript

# Define a function to perform sentiment analysis
def analyze_sentiment(text):
    # Compose the prompt
    prompt = "This is a sentiment analysis task. The goal is to determine the sentiment of the following text:\n\n" + text + "\n\nSentiment:"

    # Generate a response using the language model
    response = openai.Completion.create(
        engine="text-davinci-003",  # Choose the appropriate engine
        prompt=prompt,
        max_tokens=1,
        n=1,
        stop=None,
        temperature=0.3
    )

    # Extract the sentiment from the response
    sentiment = response.choices[0].text.strip().lower()

    return sentiment

#Generate summary

def split_text(text):
    max_chunk_size = 2048
    chunks = []
    current_chunk = ""
    for sentence in text.split("."):
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + "."
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + "."
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_summary(text):
    input_chunks = split_text(text)
    output_chunks = []
    for chunk in input_chunks:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=(f"Please summarize the following text:\n{chunk}\n\nSummary:"),
            temperature=0.5,
            max_tokens=256,
            n = 1,
            stop=None
        )
        summary = response.choices[0].text.strip()
        output_chunks.append(summary)
    return " ".join(output_chunks)

def main():
    #gcs_uri = "gs://cloud-ai-platform-hackathon23-samurais-sample-audio-mumbai/SamurAIs-Retro-Meeting-1.mp3"
    gcs_uri = "gs://ai_hackathon23_sample_audio/SamurAIs-Retro-Meeting-1.mp3"
    transcript = return_transcript(gcs_uri)
    sentiment = analyze_sentiment(transcript)
    summarized_text = generate_summary(transcript)
    output_json = {"sentiment": sentiment, "summary": summarized_text}
    return output_json
