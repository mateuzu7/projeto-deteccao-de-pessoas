# --- 1. IMPORTS NECESSÁRIOS (Isso estava faltando) ---
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import numpy as np
import cv2
import io
import PIL.Image
import os

# Certifique-se de que o cfg e o predictor já foram definidos nas células anteriores!
# Se der erro de 'cfg not defined', rode a célula de configuração do modelo antes.

# --- 2. FUNÇÕES DE SUPORTE PARA O VÍDEO ---
def array_to_image(a):
    res = PIL.Image.fromarray(a)
    byte_io = io.BytesIO()
    res.save(byte_io, format='PNG')
    return b64encode(byte_io.getvalue()).decode('ascii')

def video_stream():
  js = Javascript('''
    var video; var div = null; var stream; var captureCanvas; var imgElement; var labelElement;
    var pendingResolve = null; var shutdown = false;

    function removeDom() {
       if (stream) stream.getTracks().forEach(t => t.stop());
       if (video) video.remove();
       if (div) div.remove();
       video = null; div = null; stream = null; imgElement = null; captureCanvas = null; labelElement = null;
    }

    function onAnimationFrame() {
      if (!shutdown) window.requestAnimationFrame(onAnimationFrame);
      if (pendingResolve) {
        var result = "";
        if (!shutdown) {
          captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
          result = captureCanvas.toDataURL('image/jpeg', 0.8)
        }
        var lp = pendingResolve; pendingResolve = null; lp(result);
      }
    }

    async function createDom() {
      div = document.createElement('div');
      div.style.border = '2px solid red'; div.style.padding = '10px'; div.style.width = '660px'; div.style.background = '#000';
      labelElement = document.createElement('div');
      labelElement.innerText = "SISTEMA DE SEGURANÇA ATIVO";
      labelElement.style.color = 'white'; labelElement.style.fontWeight = 'bold';
      div.appendChild(labelElement);
      video = document.createElement('video');
      video.style.display = 'block'; video.width = 640; video.height = 480;
      div.appendChild(video);
      stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}});
      video.srcObject = stream; await video.play();
      captureCanvas = document.createElement('canvas');
      captureCanvas.width = 640; captureCanvas.height = 480;
      imgElement = document.createElement('img');
      imgElement.style.position = 'absolute'; imgElement.style.top = '40px'; imgElement.style.left = '20px';
      imgElement.style.opacity = '0.8'; div.appendChild(imgElement);
      const stopBtn = document.createElement('button');
      stopBtn.textContent = "PARAR MONITORAMENTO";
      stopBtn.onclick = () => { shutdown = true; };
      div.appendChild(stopBtn);
      document.body.appendChild(div);
      window.requestAnimationFrame(onAnimationFrame);
    }

    async function stream_frame(label, imgData) {
      if (shutdown) { removeDom(); shutdown = false; return ""; }
      if (div === null) await createDom();
      if (labelElement) labelElement.innerText = label;
      if (imgData) imgElement.src = imgData;
      return new Promise((resolve) => { pendingResolve = resolve; });
    }
    ''')
  display(js)

# --- 3. LOOP PRINCIPAL ---
# Recarrega os pesos para garantir
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

video_stream()
label_html = 'Iniciando Câmera...'
bbox_img_data = ''

try:
    while True:
        img_data = eval_js('stream_frame("{}", "{}")'.format(label_html, bbox_img_data))
        if not img_data: break

        binary = b64decode(img_data.split(',')[1])
        img = cv2.imdecode(np.frombuffer(binary, np.uint8), -1)
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")

        # Filtrar apenas classe 1 (pessoas)
        pessoas_only = instances[instances.pred_classes == 1]

        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        v = Visualizer(canvas, metadata=pessoas_metadata, scale=1.0)
        out = v.draw_instance_predictions(pessoas_only)

        bbox_img_data = 'data:image/png;base64,' + array_to_image(out.get_image())
        label_html = f"SEGURANÇA CAMPUS: {len(pessoas_only)} PESSOA(S) DETECTADA(S)"
except Exception as e:
    print("Monitoramento finalizado.")
