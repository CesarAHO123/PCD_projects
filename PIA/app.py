import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import soundfile as sf
import io

# Cargar modelo
model = tf.keras.models.load_model("modelo_emociones_rnn.keras")

# Parámetros
n_mfcc = 13
max_pad_len = 173
emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
emoji_dict = {
    'angry': '😠',
    'calm': '😌',
    'disgust': '🤢',
    'fearful': '😨',
    'happy': '😃',
    'neutral': '😐',
    'sad': '😢',
    'surprised': '😲'
}

# Extraer MFCC desde audio en memoria
def extract_mfcc_sequence(file_buffer, max_pad_len=173):
    y, sr = sf.read(file_buffer)
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return np.transpose(mfcc)  # Devuelve (173, 13)


# Configuración de la app
st.set_page_config(page_title="IA Coach Emocional", layout="centered")
st.title("¿Qué emoción expresas al hablar?")
st.markdown("Graba tu voz diciendo algo y descubre qué emoción transmite tu tono 😯")

# Grabación directa
st.header("🎤 Grabación de voz")
audio_bytes = st.audio_input("Graba una nota de voz")

if audio_bytes:
    st.success("✅ Grabación recibida")

    # Procesar audio
    audio_buffer = io.BytesIO(audio_bytes.read())
    features = extract_mfcc_sequence(audio_buffer)

    if features is not None:
        features = np.expand_dims(np.transpose(features), axis=0)


        # Predicción
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction)
        predicted_emotion = emotion_labels[predicted_index]

        # Resultado visual
        st.markdown(f"## {emoji_dict[predicted_emotion]} Emoción detectada: **{predicted_emotion.upper()}**")
        import plotly.graph_objects as go

        # Diccionario de colores por emoción
        color_dict = {
            'angry': '#E74C3C',
            'calm': '#85C1E9',
            'disgust': '#A569BD',
            'fearful': '#F4D03F',
            'happy': '#58D68D',
            'neutral': '#D5DBDB',
            'sad': '#5D6D7E',
            'surprised': '#F1948A'
        }

        # Crear gráfica con Plotly
        fig = go.Figure(data=[
            go.Bar(
                x=emotion_labels,
                y=prediction[0],
                marker_color=[color_dict[emotion] for emotion in emotion_labels]
            )
        ])

        fig.update_layout(
            title="🔍 Distribución de probabilidades por emoción",
            xaxis_title="Emoción",
            yaxis_title="Probabilidad",
            yaxis=dict(range=[0, 1]),
            template="plotly_white"
        )

        st.plotly_chart(fig)
        

