import streamlit as st
import torch 
import os 
import numpy as np
from collections import Counter
import time 
import soundfile 


st.set_page_config(page_title="Multimodal Emotion Detector", layout="wide")

# libraries that might have deeper dependencies or could be slower 
from transformers import pipeline
import whisper
import cv2
from moviepy.editor import VideoFileClip
from deepface import DeepFace

# Configuration & Model Loading
HF_DEVICE = -1 
# print(f"HuggingFace models will use device: {'CPU' if HF_DEVICE == -1 else 'GPU'}") 

@st.cache_resource 
def load_text_models():
    # print("Loading HuggingFace text models...") 
    try:
        emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
            device=HF_DEVICE
        )
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            return_all_scores=True,
            device=HF_DEVICE
        )
        print("--- HuggingFace text models loaded (console log) ---") # console logs for debugging
        return emotion_classifier, sentiment_analyzer
    except Exception as e:
        # st.error(f"Error loading HuggingFace text models: {e}") 
        print(f"CRITICAL ERROR loading HuggingFace text models: {e}") # Log to console for debugging
        return None, None

@st.cache_resource
def load_whisper_model():
    # print("Loading Whisper model...")
    try:
        
        model_root = os.path.join(os.getcwd(), ".whisper_cache") 
        os.makedirs(model_root, exist_ok=True)
        model = whisper.load_model("base", device="cpu", download_root=model_root) 
        print("--- Whisper 'base' model loaded on CPU (console log) ---")
        return model
    except Exception as e:
        print(f"CRITICAL ERROR loading Whisper model: {e}")
        return None

# Attempt to load models globally

emotion_classifier_model, sentiment_analyzer_model = load_text_models()
whisper_stt_model = load_whisper_model()

# Define Maps
sentiment_label_map = {
    'LABEL_0': 'negative',
    'LABEL_1': 'neutral',
    'LABEL_2': 'positive'
}
EMOJI_MAP = {
    "joy": "😄", "happy": "😄", 
    "sadness": "😢", "sad": "😢",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😮",
    "neutral": "😐",
    "disgust": "🤢",
    "love": "❤️" 
}
SENTIMENT_COLOR_MAP = {
    "positive": "green",
    "negative": "red",
    "neutral": "blue"
}


#  (analyze_text_emotion function - ensure it checks `if not classifier: return {"error":...}` ) 
def analyze_text_emotion(text, classifier):
    if not classifier: # This check is important
        return {"error": "Emotion classifier model not loaded."}
    if not text.strip():
        return {"error": "Input text is empty."}
    try:
        results = classifier(text)
        all_scores = results[0]
        highest_emotion = max(all_scores, key=lambda x: x['score'])
        return {
            "dominant_emotion": highest_emotion['label'],
            "dominant_emotion_confidence": highest_emotion['score'],
            "all_emotion_scores": all_scores
        }
    except Exception as e:
        return {"error": f"Error in emotion analysis: {str(e)}"}

#(analyze_text_sentiment function - ensure it checks `if not analyzer: return {"error":...}` )
def analyze_text_sentiment(text, analyzer):
    if not analyzer: # This check is important
        return {"error": "Sentiment analyzer model not loaded."}
    if not text.strip():
        return {"error": "Input text is empty."}
    try:
        results = analyzer(text)
        all_scores_raw = results[0]
        all_scores_mapped = []
        for score_item in all_scores_raw:
            mapped_label = sentiment_label_map.get(score_item['label'], score_item['label'])
            all_scores_mapped.append({'label': mapped_label, 'score': score_item['score']})
        highest_sentiment = max(all_scores_mapped, key=lambda x: x['score'])
        return {
            "dominant_sentiment": highest_sentiment['label'],
            "dominant_sentiment_confidence": highest_sentiment['score'],
            "all_sentiment_scores": all_scores_mapped
        }
    except Exception as e:
        return {"error": f"Error in sentiment analysis: {str(e)}"}

#  (analyze_text_combined function - calls the above, so it's covered)
def analyze_text_combined(text):
    # Relies on the individual functions checking for model availability
    emotion_result = analyze_text_emotion(text, emotion_classifier_model)
    sentiment_result = analyze_text_sentiment(text, sentiment_analyzer_model)

    # Check if either underlying analysis returned an error related to model loading
    if isinstance(emotion_result, dict) and "model not loaded" in emotion_result.get("error", "").lower():
        return {"error": "Emotion classifier model not loaded."}
    if isinstance(sentiment_result, dict) and "model not loaded" in sentiment_result.get("error", "").lower():
        return {"error": "Sentiment analyzer model not loaded."}
        
    if "error" in emotion_result or "error" in sentiment_result:
        return { # Propagate other errors
            "input_text": text,
            "emotion_analysis": emotion_result,
            "sentiment_analysis": sentiment_result,
            "error": f"Emotion: {emotion_result.get('error','OK')}, Sentiment: {sentiment_result.get('error','OK')}"
        }
    return {
        "input_text": text,
        "dominant_emotion": emotion_result["dominant_emotion"],
        "dominant_emotion_confidence": f"{emotion_result['dominant_emotion_confidence']:.4f}",
        "dominant_sentiment": sentiment_result["dominant_sentiment"],
        "dominant_sentiment_confidence": f"{sentiment_result['dominant_sentiment_confidence']:.4f}",
        "all_emotion_scores": emotion_result["all_emotion_scores"],
        "all_sentiment_scores": sentiment_result["all_sentiment_scores"]
    }

# (analyze_audio_from_path function - ensure it checks `if not whisper_stt_model: return {"error":...}` ) 
def analyze_audio_from_path(audio_file_path):
    if not whisper_stt_model: # This check is important
        return {"error": "Whisper (audio transcription) model not loaded."}
    if not os.path.exists(audio_file_path):
        return {"error": f"Audio file not found: {audio_file_path}"}
    try:
        print(f"Transcribing audio file: {audio_file_path} with Whisper... (console log)")
        transcription_result = whisper_stt_model.transcribe(audio_file_path, fp16=False)
        transcribed_text = transcription_result["text"]
        print(f"Whisper transcription complete: '{transcribed_text}' (console log)")

        if not transcribed_text.strip():
            return {
                "audio_file": audio_file_path,
                "transcribed_text": "(No speech detected or transcribed text is empty)",
                "analysis": "No text to analyze."
            }
        
        text_analysis_results = analyze_text_combined(transcribed_text) # This will check its own models
        return {
            "audio_file": audio_file_path,
            "transcribed_text": transcribed_text,
            "analysis": text_analysis_results
        }
    except Exception as e:
        print(f"Error in audio processing (console log): {str(e)}")
        return {"error": f"Error in audio processing: {str(e)}"}

# (extract_audio_from_video_path and analyze_video_frames_from_path as they were) 
def extract_audio_from_video_path(video_path, audio_output_path="temp_extracted_audio.wav"):
    try:
        print(f"Extracting audio from video: {video_path} (console log)...")
        with VideoFileClip(video_path) as video_clip:
            if video_clip.audio is None:
                print("No audio track found in the video (console log).")
                return None
            video_clip.audio.write_audiofile(audio_output_path, codec='pcm_s16le', logger=None) # Added logger=None
        print(f"Audio extracted and saved to {audio_output_path} (console log)")
        return audio_output_path
    except Exception as e:
        print(f"Error extracting audio from video (console log): {e}")
        if os.path.exists(audio_output_path):
            os.remove(audio_output_path)
        return None

def analyze_video_frames_from_path(video_path, frame_interval=2):
    print(f"Analyzing video frames from: {video_path} with DeepFace... (console log)")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video."}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * frame_interval) if fps > 0 else 30 * frame_interval
    
    detected_emotions_list = []
    frame_count = 0
    processed_frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frames_to_skip == 0:
            processed_frame_count += 1
            print(f"DeepFace analyzing frame {frame_count} (processed visual frame {processed_frame_count})... (console log)")
            try:
                analysis_results = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True, 
                    detector_backend='mtcnn'
                )
                emotion_to_add = None
                if isinstance(analysis_results, list) and len(analysis_results) > 0:
                    emotion_to_add = analysis_results[0]['dominant_emotion']
                elif isinstance(analysis_results, dict) and 'dominant_emotion' in analysis_results:
                    emotion_to_add = analysis_results['dominant_emotion']
                
                if emotion_to_add:
                    detected_emotions_list.append(emotion_to_add)
                    print(f"  Frame {frame_count}: Detected facial emotion - {emotion_to_add} (console log)")
                else:
                    print(f"  Frame {frame_count}: No face detected or emotion not determined (console log).")
            except Exception as e:
                print(f"  Frame {frame_count}: Error during DeepFace analysis (console log): {e}")
        frame_count += 1
    cap.release()

    if not detected_emotions_list:
        return {"most_frequent_facial_emotion": "N/A", "facial_emotion_counts": {}, "processed_frames_count": processed_frame_count}
    
    emotion_counts = Counter(detected_emotions_list)
    max_count = 0
    for emotion_value in emotion_counts.values(): # Iterate over values for max_count
        if emotion_value > max_count:
            max_count = emotion_value
    most_frequent_emotions = [emotion for emotion, count in emotion_counts.items() if count == max_count]
    
    return {
        "most_frequent_facial_emotion": ", ".join(most_frequent_emotions),
        "facial_emotion_counts": dict(emotion_counts),
        "processed_frames_count": processed_frame_count,
        "total_detected_emotions_on_frames": len(detected_emotions_list)
    }

# --- Streamlit UI (Main Body) ---
st.title("🤖 Multimodal Emotion Analysis Tool")
st.markdown("Upload text, audio, or video to detect emotions and sentiment.")


if not emotion_classifier_model or not sentiment_analyzer_model:
    st.error("Critical Error: Text analysis models failed to load. Text analysis will not work. Please check the console for details.")
if not whisper_stt_model:
    st.error("Critical Error: Whisper (audio transcription) model failed to load. Audio/Video analysis will not work. Please check console (delete cached model if checksum error).")

# Sidebar for input selection
st.sidebar.header("Input Options")
input_type = st.sidebar.selectbox("Choose input type:", ["Text", "Audio File", "Video File"])

# Placeholder for results
results_placeholder = st.empty()

#  Processing Logic (Button Clicks) 
if input_type == "Text":
    st.subheader("📝 Text Analysis")
    text_input = st.text_area("Enter text (e.g., a tweet, message, paragraph):", height=150)
    if st.button("Analyze Text", key="text_analyze_btn"):
        if not emotion_classifier_model or not sentiment_analyzer_model: 
            st.error("Text models are not available. Cannot analyze.")
        elif not text_input:
            st.warning("Please enter some text to analyze.")
        else: 
            with st.spinner("Analyzing text..."):
                results = analyze_text_combined(text_input)
            with results_placeholder.container():
                st.write("### Text Analysis Results")
                if "error" in results : 
                     st.error(f"Analysis Error: {results['error']}")
                else: 
                    main_emotion = results["dominant_emotion"]
                    main_sentiment = results["dominant_sentiment"]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label=f"Dominant Emotion {EMOJI_MAP.get(main_emotion.lower(), '')}",
                            value=main_emotion.capitalize(),
                            delta=f"Confidence: {results['dominant_emotion_confidence']}"
                        )
                    with col2:
                        st.metric(
                            label=f"Dominant Sentiment",
                            value=main_sentiment.capitalize(),
                            delta=f"Confidence: {results['dominant_sentiment_confidence']}"
                        )
                        st.markdown(f"<p style='color:{SENTIMENT_COLOR_MAP.get(main_sentiment, 'black')};'>Sentiment is {main_sentiment}</p>", unsafe_allow_html=True)
                    with st.expander("Show all emotion scores"):
                        st.json(results["all_emotion_scores"])
                    with st.expander("Show all sentiment scores"):
                        st.json(results["all_sentiment_scores"])
        
elif input_type == "Audio File":
    st.subheader("🎵 Audio Analysis")
    uploaded_audio_file = st.file_uploader("Upload an audio file (MP3, WAV):", type=["mp3", "wav", "m4a", "flac"])
    if st.button("Analyze Audio", key="audio_analyze_btn"):
        if not whisper_stt_model: 
            st.error("Audio transcription model (Whisper) is not available. Cannot analyze.")
        elif not uploaded_audio_file:
            st.warning("Please upload an audio file.")
        else: 
            with st.spinner(f"Processing audio file: {uploaded_audio_file.name}... (This may take a while for transcription)"):
                temp_dir = "temp_uploads"
                os.makedirs(temp_dir, exist_ok=True)
                # Sanitize filename slightly for safety, though os.path.join handles paths
                base, ext = os.path.splitext(uploaded_audio_file.name)
                safe_filename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in base) + ext
                temp_audio_path = os.path.join(temp_dir, safe_filename)
                
                with open(temp_audio_path, "wb") as f:
                    f.write(uploaded_audio_file.getbuffer())
                
                audio_results = analyze_audio_from_path(temp_audio_path)
                
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)

            with results_placeholder.container():
                st.write("### Audio Analysis Results")
                if "error" in audio_results:
                    st.error(audio_results["error"])
                else:
                    st.write(f"**Transcribed Text:** {audio_results['transcribed_text']}")
                    text_analysis = audio_results['analysis']
                    if isinstance(text_analysis, str): 
                         st.info(text_analysis)
                    elif "error" in text_analysis: # Check for error from analyze_text_combined
                        st.error(f"Analysis Error (from transcript): {text_analysis['error']}")
                    else: 
                        main_emotion = text_analysis["dominant_emotion"]
                        main_sentiment = text_analysis["dominant_sentiment"]
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                label=f"Dominant Emotion (from text) {EMOJI_MAP.get(main_emotion.lower(), '')}",
                                value=main_emotion.capitalize(),
                                delta=f"Confidence: {text_analysis['dominant_emotion_confidence']}"
                            )
                        with col2:
                             st.metric(
                                label=f"Dominant Sentiment (from text)",
                                value=main_sentiment.capitalize(),
                                delta=f"Confidence: {text_analysis['dominant_sentiment_confidence']}"
                            )
                             st.markdown(f"<p style='color:{SENTIMENT_COLOR_MAP.get(main_sentiment, 'black')};'>Sentiment is {main_sentiment}</p>", unsafe_allow_html=True)
                        with st.expander("Show all emotion scores (from text)"):
                            st.json(text_analysis["all_emotion_scores"])
                        with st.expander("Show all sentiment scores (from text)"):
                            st.json(text_analysis["all_sentiment_scores"])

elif input_type == "Video File":
    st.subheader("🎬 Video Analysis")
    uploaded_video_file = st.file_uploader("Upload a video file (MP4 recommended):", type=["mp4", "mov", "avi", "mkv"])
    frame_proc_interval = st.slider(
        "Video frame analysis interval (seconds):", 
        min_value=0.5, max_value=5.0, value=2.0, step=0.5,
        help="Lower values analyze more frames (more detail, but much slower). E.g., 1.0 means 1 frame per second."
    )

    if st.button("Analyze Video", key="video_analyze_btn"):
        if not whisper_stt_model and (not emotion_classifier_model or not sentiment_analyzer_model) : # Basic check
             st.error("Core models (Whisper and/or Text models) are not available. Video analysis may be incomplete or fail.")
        elif not uploaded_video_file:
            st.warning("Please upload a video file.")
        else:
            with st.spinner(f"Processing video file: {uploaded_video_file.name}... (This will take significant time!)"):
                temp_dir = "temp_uploads"
                os.makedirs(temp_dir, exist_ok=True)
                base, ext = os.path.splitext(uploaded_video_file.name)
                safe_filename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in base) + ext
                temp_video_path = os.path.join(temp_dir, safe_filename)
                
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_video_file.getbuffer())

                audio_analysis_report = {"transcribed_text": "N/A", "analysis": "Audio processing skipped or failed."}
                # Ensure unique extracted audio filename to prevent clashes if multiple video analyses run
                temp_extracted_audio_filename = f"temp_ext_audio_{int(time.time())}_{safe_filename}.wav"
                temp_extracted_audio_path = os.path.join(temp_dir, temp_extracted_audio_filename)

                if whisper_stt_model: # Only attempt if model is loaded
                    extracted_audio_path_from_video = extract_audio_from_video_path(temp_video_path, audio_output_path=temp_extracted_audio_path)
                    if extracted_audio_path_from_video:
                        audio_analysis_report = analyze_audio_from_path(extracted_audio_path_from_video)
                        if os.path.exists(extracted_audio_path_from_video):
                            os.remove(extracted_audio_path_from_video)
                else:
                     audio_analysis_report = {"error": "Whisper model not loaded, skipping audio analysis from video."}
                
                facial_emotion_report = analyze_video_frames_from_path(temp_video_path, frame_interval=frame_proc_interval)

                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
            
            with results_placeholder.container():
                st.write("### Video Analysis Results")
                st.write("#### Audio Content Analysis (from video)")
                if "error" in audio_analysis_report:
                    st.error(f"Audio Analysis Error: {audio_analysis_report['error']}")
                else:
                    st.write(f"**Transcribed Text:** {audio_analysis_report.get('transcribed_text', 'N/A')}")
                    text_analysis = audio_analysis_report.get('analysis', {})
                    if isinstance(text_analysis, str):
                         st.info(text_analysis)
                    elif "error" in text_analysis:
                        st.error(f"Text Analysis Error (from audio): {text_analysis['error']}")
                    else:
                        main_emotion_audio = text_analysis["dominant_emotion"]
                        main_sentiment_audio = text_analysis["dominant_sentiment"]
                        col1, col2 = st.columns(2)
                        with col1:
                             st.metric(
                                label=f"Dominant Emotion (from audio) {EMOJI_MAP.get(main_emotion_audio.lower(), '')}",
                                value=main_emotion_audio.capitalize(),
                                delta=f"Confidence: {text_analysis['dominant_emotion_confidence']}"
                            )
                        with col2:
                            st.metric(
                                label=f"Dominant Sentiment (from audio)",
                                value=main_sentiment_audio.capitalize(),
                                delta=f"Confidence: {text_analysis['dominant_sentiment_confidence']}"
                            )
                            st.markdown(f"<p style='color:{SENTIMENT_COLOR_MAP.get(main_sentiment_audio, 'black')};'>Sentiment (from audio) is {main_sentiment_audio}</p>", unsafe_allow_html=True)
                        with st.expander("Show all emotion scores (from audio text)"):
                            st.json(text_analysis["all_emotion_scores"])

                st.write("#### Facial Expression Analysis (from video frames)")
                if "error" in facial_emotion_report:
                    st.error(f"Facial Analysis Error: {facial_emotion_report['error']}")
                else:
                    # Handle cases where most_frequent_facial_emotion might be "N/A" or multiple
                    mf_emotion_str = facial_emotion_report.get('most_frequent_facial_emotion', 'N/A')
                    first_mf_emotion = mf_emotion_str.split(',')[0].strip().lower() if mf_emotion_str != 'N/A' else ''
                    
                    st.metric(
                        label=f"Most Frequent Facial Emotion {EMOJI_MAP.get(first_mf_emotion, '')}", 
                        value=mf_emotion_str
                    )
                    st.write(f"**Facial Emotion Counts:** {facial_emotion_report.get('facial_emotion_counts', {})}")
                    st.caption(f"Frames processed for facial analysis: {facial_emotion_report.get('processed_frames_count', 'N/A')}, Total facial emotions detected: {facial_emotion_report.get('total_detected_emotions_on_frames', 'N/A')}")

st.markdown("---")
st.markdown("Built by Prakash")
st.markdown(f"First analysis of each type may take some time as models are loaded.")
