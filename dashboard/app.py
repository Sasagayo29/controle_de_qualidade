import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image  # Pillow
from pathlib import Path

# Importar a função de pré-processamento específica do MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Configuração da Página ---
st.set_page_config(
    page_title="Controle de Qualidade Visual",
    page_icon="🤖",
    layout="wide"
)

# --- Constantes ---
# O tamanho que o modelo espera
IMG_SIZE = (224, 224)
# As classes (na ordem que o Keras aprendeu na Célula 1)
CLASS_NAMES = ['def_front', 'ok_front']

# --- Carregamento do Modelo (com cache) ---


@st.cache_resource
def load_keras_model():
    """Carrega o modelo .h5 treinado."""
    try:
        # Caminhos (usando pathlib)
        SCRIPT_DIR = Path(__file__).resolve().parent
        PROJECT_ROOT = SCRIPT_DIR.parent
        MODEL_PATH = PROJECT_ROOT / "models" / "quality_control_model.h5"

        print(f"Carregando modelo de: {MODEL_PATH}")

        # Carregar o modelo
        # compile=False é usado para acelerar o carregamento,
        # já que não vamos treinar.
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        print("Modelo carregado com sucesso.")
        return model
    except FileNotFoundError:
        st.error(
            f"Erro: Arquivo 'quality_control_model.h5' não encontrado na pasta 'models/'.")
        st.error(
            "Por favor, re-execute o notebook de treinamento (Célula 2) para salvar o modelo corrigido.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None


model = load_keras_model()


def process_image(image_pil):
    """Converte a imagem (PIL) para o formato que o modelo espera."""

    # 1. Converter para 3 canais (RGB)
    # O dataset original é grayscale, mas o MobileNetV2 espera 3 canais.
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")

    # 2. Redimensionar para 224x224
    image_resized = image_pil.resize(IMG_SIZE)

    # 3. Converter para array numpy
    image_array = np.array(image_resized)

    # 4. Adicionar dimensão de 'batch' (o modelo espera 4D)
    image_batch = np.expand_dims(image_array, axis=0)

    # 5. Aplicar o pré-processamento do MobileNetV2 (normalizar pixels para [-1, 1])
    # Esta etapa foi movida do modelo (no notebook) para cá (no app).
    image_preprocessed = preprocess_input(image_batch)

    return image_preprocessed


# --- Interface do Usuário (UI) ---
st.title("🏭 Sistema de Controle de Qualidade Visual (CNN)")
st.markdown("Faça o upload de uma imagem de uma peça fundida. O modelo irá classificá-la como 'Aprovado' ou 'Defeituoso'.")

if model is not None:
    # 1. Widget de Upload
    uploaded_file = st.file_uploader(
        "Escolha uma imagem de peça...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # 2. Carregar e Mostrar Imagem
        image_pil = Image.open(uploaded_file)

        st.image(image_pil, caption="Imagem Carregada",
                 use_column_width=True, width=300)

        # 3. Processar Imagem e Fazer Previsão
        with st.spinner("Analisando imagem... 🧠"):

            # Converter a imagem para o formato do modelo
            image_for_model = process_image(image_pil)

            # Fazer a previsão
            prediction_probs = model.predict(image_for_model)

            # A saída é 'sigmoid', um float entre 0.0 e 1.0
            prediction_prob = prediction_probs[0][0]

            # 4. Obter o Veredito
            if prediction_prob > 0.5:
                # Classe 1 ('ok_front')
                verdict_index = 1
                confidence = prediction_prob
            else:
                # Classe 0 ('def_front')
                verdict_index = 0
                confidence = 1 - prediction_prob

            verdict = CLASS_NAMES[verdict_index]

        # 5. Mostrar o Resultado
        st.markdown("---")
        if verdict == 'ok_front':
            st.success(f"## ✅ Veredito: APROVADO")
            st.metric(label="Confiança do Modelo",
                      value=f"{confidence*100:.2f}%")
        else:
            st.error(f"## ❌ Veredito: DEFEITUOSO")
            st.metric(label="Confiança do Modelo",
                      value=f"{confidence*100:.2f}%")

        st.info(
            f"Classe prevista: {verdict} (Probabilidade: {prediction_prob:.4f})")

else:
    st.error("Modelo não pôde ser carregado. O dashboard não pode funcionar.")
