import streamlit as st
import cv2
import numpy as np
import av
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp

# Configuração da página
icon = Image.open("img/icon.png")
st.set_page_config(page_title="Idom+ Filtros Avançados", page_icon=icon)

# Filtros disponíveis
filter_options = {
    "Óculos 1": ("filters/filter16.png", "olhos"),
    "Óculos 2": ("filters/filter3.png", "olhos"),
    "Máscara completa": ("filters/filter3.png", "face"),
    "Chapéu de festa": ("filters/filter4.png", "cabeca"),
}

def load_filter(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def apply_filter(image, landmarks, filtro_img, tipo):
    if filtro_img is None:
        return image

    ih, iw = image.shape[:2]

    def mp_point(idx):
        pt = landmarks[idx]
        return np.array([int(pt.x * iw), int(pt.y * ih)])

    if tipo == "olhos":
        left_eye = mp_point(33)
        right_eye = mp_point(263)

        eye_width = np.linalg.norm(right_eye - left_eye)
        scale_factor = 2.3
        filter_width = int(eye_width * scale_factor)
        filter_height = int(filter_width * filtro_img.shape[0] / filtro_img.shape[1])

        center = ((left_eye + right_eye) // 2).astype(int)
        top_left = np.array([center[0] - filter_width // 2, center[1] - filter_height // 2])

        # Redimensionar o filtro
        resized_filter = cv2.resize(filtro_img, (filter_width, filter_height), interpolation=cv2.INTER_AREA)

        x1, y1 = top_left
        x2, y2 = x1 + filter_width, y1 + filter_height

        # Calcular as áreas de sobreposição (corte se fora dos limites da imagem)
        filter_x1 = max(0, -x1)
        filter_y1 = max(0, -y1)
        filter_x2 = filter_width - max(0, x2 - iw)
        filter_y2 = filter_height - max(0, y2 - ih)

        img_x1 = max(x1, 0)
        img_y1 = max(y1, 0)
        img_x2 = min(x2, iw)
        img_y2 = min(y2, ih)

        # Verifica se a ROI é válida
        if filter_x1 >= filter_x2 or filter_y1 >= filter_y2:
            return image

        filter_roi = resized_filter[filter_y1:filter_y2, filter_x1:filter_x2]
        overlay_color = filter_roi[:, :, :3]
        mask = filter_roi[:, :, 3:] / 255.0

        roi = image[img_y1:img_y2, img_x1:img_x2]
        image[img_y1:img_y2, img_x1:img_x2] = (1 - mask) * roi + mask * overlay_color

    return image



class FilterTransformer(VideoTransformerBase):
    def __init__(self, filtro_path, tipo):
        self.filtro_img = load_filter(filtro_path)
        self.tipo = tipo
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            img = apply_filter(img, landmarks, self.filtro_img, self.tipo)

        return img

def main():
    st.title("Trabalho Estácio")
    st.sidebar.subheader("Escolha um filtro facial:")

    filtro_nome = st.sidebar.selectbox("Filtro:", list(filter_options.keys()))
    filtro_path, tipo = filter_options[filtro_nome]

    webrtc_streamer(
        key="filtros-homografia",
        video_processor_factory=lambda: FilterTransformer(filtro_path, tipo),
        async_processing=True,
    )

if __name__ == "__main__":
    main()
