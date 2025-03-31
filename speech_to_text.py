from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2")
result = asr("test2.mp3", return_timestamps=True)
print(result)