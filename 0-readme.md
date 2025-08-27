cd /mnt/d/Desktop/mq && conda activate /root/py310
mineru-api --host 0.0.0.0 --port 8815 --device cuda --source local
streamlit run app.py --server.port 8501


cpolar http 8501

Qwen/Qwen2.5-72B-Instruct
qwen3-235b-a22b

下载mineru全部模型到本地
mineru-models-download
modelscope

模型源设置
export MINERU_MODEL_SOURCE=modelscope

auto不加速的效果似乎比加速的效果好（不加速解析结果中段内分行少于加速解析）
不加速： mineru -p <input_path> -o <output_path>  --source local
加速：   mineru -p <input_path> -o <output_path>  --source local  -b vlm-transformers -t false -l ch

DocAnalysis init, this may take some times. model init cost: 1623.6848361492157



语义向量模型
BAAI/bge-m3
intfloat/e5-mistral-7b-instruct
Alibaba-NLP/gte-Qwen2-7B-instruct


mineru-api --host 0.0.0.0 --port 8815 --device cuda --source local


以下是一些来自同一个类别但是可能不同语境的句子，请基于它们的整体内容，从客观、语言分析的角度出发，提炼一个简洁的一句话总结，可包含事件背景（起因）、主要过程（经过）、以及最后的变化或影响（结果）。无需评论立场，请仅描述现象与事实：

python recursive_cluster_v10.py --input output-cluster-step3-v10.json --output output-cluster-step3-v10-recursive_cluster.json 

python recursive_cluster_v10.py --input all_paragraph_texts.json --output output-cluster-step3-v10-recursive_cluster-all.json 

python cluster-v8.py --n_clusters 9 





mineru.cli.models_download:download_pipeline_models:73 - Downloading model: models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt


mineru.cli.models_download:download_pipeline_models:73 - Downloading model: models/MFD/YOLO/yolo_v8_ft.pt


mineru.cli.models_download:download_pipeline_models:73 - Downloading model: models/MFR/unimernet_hf_small_2503

Downloading Model from https://www.modelscope.cn to directory: /root/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1.0

mineru.cli.models_download:download_pipeline_models:73 - Downloading model: models/OCR/paddleocr_torch

mineru.cli.models_download:download_pipeline_models:73 - Downloading model: models/ReadingOrder/layout_reader

mineru.cli.models_download:download_pipeline_models:73 - Downloading model: models/TabRec/SlanetPlus/slanet-plus.onnx

mineru.cli.models_download:download_pipeline_models:75 - Pipeline models downloaded successfully to: /root/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1___0

Downloading Model from https://www.modelscope.cn to directory: /root/.cache/modelscope/hub/models/OpenDataLab/MinerU2.0-2505-0.9B


