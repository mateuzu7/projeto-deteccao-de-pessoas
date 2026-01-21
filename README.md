# projeto-deteccao-de-pessoas
Sistema de IA para detecção de pessoas na sala de estudos do Campus da UFC de Itapajé

# Sistema de Detecção de Pessoas em Ambientes do Campus

Este projeto apresenta uma solução de Visão Computacional baseada em Redes Neurais Convolucionais (CNN) para a detecção automática de pessoas. O sistema foi desenvolvido como requisito final da disciplina de Inteligência Artificial (2025.2), utilizando o framework **Detectron2**.

## 🎯 Objetivo e Aplicação em Segurança da Informação

O objetivo principal é monitorar ambientes reais do Campus (laboratórios, corredores e salas de aula) para apoiar a segurança patrimonial e física.

**Aplicações em Segurança:**
1.  **Monitoramento de Perímetro:** Detecção de intrusão em áreas restritas (ex: laboratórios de servidores) fora do horário comercial.
2.  **Análise de Ocupação:** Controle de lotação em tempo real para conformidade com normas de segurança (evacuação) e prevenção de aglomerações.
3.  **Auditoria de Acesso:** Registro visual automatizado de entradas e saídas sem necessidade de intervenção humana constante.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Framework:** Detectron2 (Facebook AI Research)
* **Modelo Base:** Mask R-CNN (ResNet-50-FPN) pré-treinado no dataset COCO.
* **Técnica:** Transfer Learning (Fine-Tuning) para a classe `pessoas`.
* **Rotulagem:** Roboflow (Formato COCO JSON).

## 📊 Metodologia

1.  **Coleta de Dados:** Foram capturadas ~94 imagens em ambientes do Campus, variando iluminação e ângulos.
2.  **Rotulagem:** Anotação manual utilizando Polígonos (Segmentação de Instâncias) para delimitar precisamente o contorno das pessoas. Esta abordagem permite que o modelo aprenda não apenas a localização (Bounding Box), mas a forma exata dos indivíduos nos ambientes do Campus.
3.  **Treinamento:**
    * **Iterações:** 1000
    * **Learning Rate:** 0.00025
    * **Batch Size:** 2
    * **Num Classes:** 2 (Mapeamento ajustado para compatibilidade com Roboflow).

## 📈 Resultados e Métricas

O modelo alcançou resultados expressivos para o dataset de teste:

| Métrica | Valor | Interpretação |
| :--- | :--- | :--- |
| **mAP (IoU=0.50:0.95)** | **67.1%** | Alta precisão geral na detecção. |
| **AP50 (IoU=0.50)** | **87.8%** | O modelo detecta corretamente a presença humana em quase 88% dos casos. |
| **AP75** | **81.4%** | Alta fidelidade no ajuste da caixa delimitadora. |

### Exemplos Visuais

**1. Detecção em Imagem Estática:**
![Exemplo de Detecção](results/images/testenovo.png)


**2. Monitoramento em Tempo Real (Webcam):**
O sistema é capaz de realizar inferência em vídeo, simulando uma câmera de segurança IP.
![Webcam Demo](results/images/detect20-01)


## 🚀 Como Executar

⚠️ Importante:
Este projeto necessita de GPU para treinamento e inferência.
Execute exclusivamente no Google Colab com GPU ativada.

🔧 1. Configurar o Ambiente no Google Colab

---
```
# 3. Passo a Passo Para Execução

## 🛠️ Preparando o Ambiente e Instalando o Detectron2

Inicialmente, estando no ambiente de nuvem (Google Colab), altere o ambiente de execução para **GPU**.
Depois, verifique a existência e o status da GPU executando a célula abaixo:

```bash
!nvidia-smi

```

Se bem-sucedida, você verá uma tabela mostrando a GPU (ex: Tesla T4).

Em seguida, adicione o arquivo zipado do seu dataset (exportado do Roboflow) ao diretório `/content` do ambiente e execute o comando exato abaixo para descompactar (note o uso de aspas devido aos espaços no nome):


```
!unzip "Detect.v1-roboflow-instant-1--eval-.coco (1).zip"

```

Instale a versão estável do Detectron2 compatível com o Colab:


```
!python -m pip install 'git+[https://github.com/facebookresearch/detectron2.git](https://github.com/facebookresearch/detectron2.git)'

```

* * * * *

⚙️ Configuração do Dataset e Treinamento do Modelo
--------------------------------------------------

Nesta etapa, o código realiza o registro dos datasets (`train`, `valid`, `test`) no formato COCO. Em seguida, prepara o modelo **Mask R-CNN** (ResNet-50-FPN) usando o Detectron2.

Diferente do Faster R-CNN, este modelo é capaz de segmentação, mas aqui estamos focando na detecção. Definimos **2 classes** (0: objects, 1: pessoas) e configuramos os hiperparâmetros de treino.

Arquivo: `/projeto-deteccao-pessoas/training/train.py`


```python
import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import os, cv2, random
from google.colab.patches import cv2_imshow
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# Registrando os datasets (usando os nomes das pastas que o zip criou)
try:
    register_coco_instances("pessoas_train", {}, "/content/train/_annotations.coco.json", "/content/train")
    register_coco_instances("pessoas_valid", {}, "/content/valid/_annotations.coco.json", "/content/valid")
    register_coco_instances("pessoas_test", {}, "/content/test/_annotations.coco.json", "/content/test")
except:
    print("Datasets já registrados ou erro nos caminhos.")

pessoas_metadata = MetadataCatalog.get("pessoas_train")

# Configuração do Modelo Mask R-CNN
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("pessoas_train",)
cfg.DATASETS.TEST = ("pessoas_valid",) # Validação durante o treino
cfg.DATALOADER.NUM_WORKERS = 2

# Pesos iniciais (Transfer Learning)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000 # Quantidade ajustada para o dataset
cfg.SOLVER.STEPS = []

# DEFINIÇÃO DE CLASSES: 2 (0: objects, 1: pessoas)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

cfg.OUTPUT_DIR = "/content/output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Iniciar Treinamento
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

```

* * * * *

🔍 Inferência e Visualização
----------------------------

Aqui, o código carrega os pesos treinados (`model_final.pth`) e realiza inferência em:

1.  Imagens aleatórias do conjunto de teste.

2.  Uma imagem externa específica (ex: foto do WhatsApp).

O script filtra especificamente a **classe 1 ("pessoas")**, ignorando outros objetos, e exibe o resultado com fundo preto e branco (`ColorMode.IMAGE_BW`) para destacar a detecção.

Arquivo: `/projeto-deteccao-pessoas/inference/test_model.py`



```python
import os, cv2, random
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from google.colab.patches import cv2_imshow

# Carregar o modelo treinado
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Confiança mínima
predictor = DefaultPredictor(cfg)

print("--- TESTE 1: IMAGENS ALEATÓRIAS ---")
dataset_dicts = DatasetCatalog.get("pessoas_test")
for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)

    # Filtramos para mostrar apenas a classe 1 (pessoas)
    instances = outputs["instances"].to("cpu")
    mask = instances.pred_classes == 1
    pessoas_only = instances[mask]

    v = Visualizer(im[:, :, ::-1],
                   metadata=pessoas_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW # Fundo PB destaca a detecção
    )
    out = v.draw_instance_predictions(pessoas_only)
    print(f"Resultado para: {d['file_name']}")
    cv2_imshow(out.get_image()[:, :, ::-1])

print("\n--- TESTE 2: IMAGEM EXTERNA ---")
# Caminho da sua nova imagem
caminho_imagem_nova = "/content/WhatsApp Image 2026-01-18 at 11.06.07 PM.jpeg"

if os.path.exists(caminho_imagem_nova):
    im = cv2.imread(caminho_imagem_nova)
    outputs = predictor(im)

    # Filtrar classe 1 (pessoas)
    instances = outputs["instances"].to("cpu")
    pessoas_only = instances[instances.pred_classes == 1]

    v = Visualizer(im[:, :, ::-1],
                   metadata=pessoas_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW)

    out = v.draw_instance_predictions(pessoas_only)
    cv2_imshow(out.get_image()[:, :, ::-1])
else:
    print("Imagem externa não encontrada. Verifique o caminho.")

```

* * * * *

📊 Avaliação de Desempenho (Métricas COCO)
------------------------------------------

O sistema realiza a avaliação quantitativa utilizando o `COCOEvaluator`. Configuramos `tasks=("bbox",)` para avaliar apenas as caixas delimitadoras, evitando erros relacionados à segmentação (máscaras) caso o dataset não esteja perfeitamente rotulado para tal.

Arquivo: `/projeto-deteccao-pessoas/results/metrics/evaluation.py`

```python
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Avaliamos apenas BBOX (caixas) para evitar erro de segmentação
evaluator = COCOEvaluator("pessoas_test", output_dir="./output", tasks=("bbox",))
val_loader = build_detection_test_loader(cfg, "pessoas_test")

print("--- MÉTRICAS DE DESEMPENHO ---")
results = inference_on_dataset(predictor.model, val_loader, evaluator)
print(results)

```

* * * * *

📹 Monitoramento em Tempo Real (Webcam)
---------------------------------------

Essa etapa injeta código JavaScript no Colab para acessar a webcam do navegador. O Python processa cada frame, detecta a classe "pessoas" e desenha as caixas em tempo real.

Arquivo: `/projeto-deteccao-pessoas/inference/webcam_monitoring.py`



```python
# --- 1. IMPORTS NECESSÁRIOS ---
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import numpy as np
import cv2
import io
import PIL.Image
import os

# (Assume que cfg e predictor já estão carregados das etapas anteriores)

# --- 2. FUNÇÕES DE SUPORTE JS/PYTHON ---
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
```

## 📁 Estrutura do Repositório

* `data/`: Amostras do dataset e anotações.
* `training/`: Scripts de configuração e treinamento (Fine-tuning).
* `inference/`: Scripts para teste em imagens e webcam.
* `results/`: Gráficos de métricas e evidências visuais.
* `model/`: (Link para download do modelo .pth).

---
**Autor:** Mateus Oliveira
**Disciplina:** Inteligência Artificial - 2025.2
