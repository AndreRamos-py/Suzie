#Out main file.

import speech_recognition as sr

# Create a recognizer
r = sr.Recognizer()

# Open the microphone
with sr.Microphone() as source:
    while True:
        audio = r.listen(source) # Sets the microphone as the audio source

        print(r.recognize_google(audio, language="pt-br"))
