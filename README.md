# Multimodal Emotion Detector

## Project Overview

This project is a smart emotion analysis tool built with Python and Streamlit. It accepts text, audio, or video input and performs emotion and sentiment analysis, displaying the results with confidence scores and visual feedback.

This application was developed as an important assignment, aiming to explore and implement multimodal emotion recognition techniques.

## Features

**Multimodal Input:** Accepts text, audio (MP3/WAV), and video (MP4) files.
**Emotion Detection:** Identifies emotional tones such as Happy, Angry, Sad, Neutral, Surprise, Fear, Disgust.
**Sentiment Analysis:** Classifies input as Positive, Negative, or Neutral.
**Confidence Scores:** Displays the confidence level for detected emotions and sentiments.
**Visual Feedback:** Uses emojis and color-coding for intuitive understanding of results.
**Transcription:** Transcribes speech from audio and video inputs.
**Facial Emotion Recognition:** Detects emotions from facial expressions in video frames.

## Technologies Used

**Python:** Core programming language.
**Streamlit:** For building the interactive web application.
**Hugging Face Transformers:**
    *`j-hartmann/emotion-english-distilroberta-base`: For text-based emotion classification.
    *   `cardiffnlp/twitter-roberta-base-sentiment-latest`: For text-based sentiment analysis.
**OpenAI Whisper:** For speech-to-text (audio transcription).
**DeepFace:** For facial expression recognition from video frames.
**MoviePy:** For audio extraction from video files.
**OpenCV:** For video frame processing.

## Setup and Installation (Local Execution)

Follow these steps to set up and run the project locally:

1.**Clone the Repository (Optional if you have the files):**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```

2.**Create a Python Virtual Environment:**
    It's highly recommended to use a virtual environment. This project was developed and tested with Python 3.10.
    ```bash
    # For Windows, using Python 3.10 via py.exe launcher
    py -3.10 -m venv venv
    # For macOS/Linux
    # python3.10 -m venv venv

    # Activate the environment
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    # source venv/bin/activate
    ```

3.  **Install FFmpeg:**
    This application requires FFmpeg for audio and video processing. Ensure it's installed on your system and accessible via the system PATH.
    **Windows:** Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and add the `bin` folder to your PATH.
    **macOS (using Homebrew):** `brew install ffmpeg`
    *   **Linux (using apt):** `sudo apt update && sudo apt install ffmpeg`
    Verify by typing `ffmpeg -version` in your terminal.

4.**Install Python Dependencies:**
    With the virtual environment activated, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure your `requirements.txt` file is up-to-date and correctly specifies the CPU version of PyTorch, e.g., `torch --index-url https://download.pytorch.org/whl/cpu`)*

5.**Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    The application should open in your default web browser (usually at `http://localhost:8501`).

## How to Use

1.Once the application is running, select the input type from the sidebar: "Text," "Audio File," or "Video File."
2.  **For Text:** Enter the text in the provided text area and click "Analyze Text."
3.  **For Audio File:** Upload an MP3 or WAV file containing speech and click "Analyze Audio."
4.  **For Video File:** Upload an MP4 video file (preferably with a visible face and audible speech). You can adjust the "frame analysis interval" slider to balance detail vs. processing time. Click "Analyze Video."
5.  The analysis results, including detected emotions, sentiment, confidence scores, and any transcribed text, will be displayed.

**Note on Performance:**
The application is configured to run on CPU.
Model loading (especially for Whisper and DeepFace) and analysis of audio/video files can be time-consuming on CPU. Please be patient during these operations.

## File Structure

 .
├── app.py # Main Streamlit application script
├── requirements.txt # Python dependencies
├── README.md # This file
├── .gitignore # Specifies intentionally untracked files
├── temp_uploads/ # (Created at runtime for temporary file storage, should be in .gitignore)
└── .whisper_cache/ # (Created by Whisper for model caching, should be in .gitignore)

## Potential Issues & Troubleshooting

**Model Download Failures (`getaddrinfo failed`, Checksum Errors):**
    Ensure a stable internet connection.
    Check firewall/antivirus settings; they might be blocking downloads. Consider adding an exclusion for Python or the project folder if issues persist.
    *   For checksum errors with Whisper, delete the cached model files (usually in `C:\Users\YOUR_USERNAME\.cache\whisper\` or the project's `.whisper_cache/` if `download_root` was used) to force a fresh download.
**`ModuleNotFoundError`:** Ensure your virtual environment is active and all packages from `requirements.txt` are installed correctly.
**Old `moviepy` version installing:** If `pip install moviepy` installs an old version (e.g., 2.2.1), try `pip install moviepy==1.0.3` to force the correct version.
**TensorFlow/Keras errors (`tf-keras` missing):** Run `pip install tf-keras` if you encounter errors related to TensorFlow 2.16+ and Keras.

## Future Enhancements (Ideas)

GPU support for faster processing.
Option to select different models for each modality.
More detailed visualization of emotion timelines for audio/video.
Deployment to a more robust cloud platform.

## Acknowledgements

This project utilizes several powerful open-source libraries and pre-trained models. Thanks to the developers and communities behind:
Streamlit
Hugging Face
OpenAI (for Whisper)
DeepFace
MoviePy, OpenCV, and others.

## **Author:** Prakash Sharma