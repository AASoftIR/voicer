import streamlit as st
import whisper
from pydub import AudioSegment
from pydub.effects import normalize, high_pass_filter
import os

def convert_audio_to_wav(input_file, output_file):
    """
    Converts an audio file to WAV format using pydub.
    """
    audio = AudioSegment.from_file(input_file)
    audio = normalize(audio)  # Normalize volume
    audio = high_pass_filter(audio, cutoff=200)  # Reduce low-frequency noise
    audio.export(output_file, format="wav")
    return output_file

def transcribe_audio(file_path, language="fa"):
    """
    Transcribes Persian audio to text using Whisper.
    """
    model = whisper.load_model("small")  # 'base' for faster, 'medium' or 'large' for better accuracy.
    result = model.transcribe(file_path, language=language)
    return result["text"]

def main():
    st.title("Persian Voice-to-Text Transcription")
    st.write("Upload a WAV file, and this app will transcribe it into Persian text.")
    
    # File uploader restricted to WAV files
    uploaded_file = st.file_uploader("Upload your audio file (WAV only)", type=["wav"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav", start_time=0)
        
        with st.spinner("Processing audio..."):
            try:
                # Save the uploaded file temporarily
                input_path = "uploaded_audio.wav"
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Transcribe the WAV audio directly
                transcription = transcribe_audio(input_path, language="fa")
                
                st.success("Transcription Complete!")
                st.text_area("Transcribed Text (Persian)", transcription, height=300)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

            finally:
                # Cleanup temporary files
                if os.path.exists(input_path):
                    os.remove(input_path)

if __name__ == "__main__":
    main()
