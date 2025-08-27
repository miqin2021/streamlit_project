# cluster-v8.py
import os
import json
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
import argparse  
import time  # 新增：用于重试延迟
OUTPUT_DIR = "data/outputs"
OUTPUT_DIR_4 = os.path.join(OUTPUT_DIR, "step4")

# ==== 配置参数 ====
JSON_DIR = "data/json_layout"
OUTPUT_FILE = os.path.join(OUTPUT_DIR_4, "output-cluster-step4.json")
BAD_PARAGRAPH_LOG = os.path.join(OUTPUT_DIR_4, "output-cluster-step4-ignored-paragraphs.txt")
USE_TFIDF_KEYWORDS = True
ALL_TEXTS_JSON = os.path.join(OUTPUT_DIR_4, "all_paragraph_texts.json")

CONFIG_FILE = "llm.config"

# === 关键修改：更新命令行参数解析，添加--selected_files ===
def parse_args():
    parser = argparse.ArgumentParser(description="执行文本聚类")
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=9,
        help="聚类数量，即要分成几类 (默认: 9)"
    )
    # 新增：接收选中的JSON文件名（逗号分隔）
    parser.add_argument(
        "--selected_files",
        type=str,
        required=True,
        help="Step3选中的JSON文件名，用英文逗号分隔"
    )
    return parser.parse_args()

# 读取命令行参数
args = parse_args()
N_CLUSTERS = args.n_clusters  
# 关键修改：解析选中的文件列表（逗号分隔→列表）
SELECTED_JSON_FILES = args.selected_files.split(",")
print(f"📌 目标类别数：{N_CLUSTERS}")
print(f"📌 待处理文件数：{len(SELECTED_JSON_FILES)}")
for idx, file in enumerate(SELECTED_JSON_FILES, 1):
    print(f"  {idx}. {file}")

# ==== OpenAI / ModelScope 配置（原有代码不变） ====
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    llm_config = json.load(f)

client = OpenAI(
    base_url=llm_config["base_url"],
    api_key=llm_config["api_key"]
)
LLM_MODEL = llm_config.get("model_name", "Qwen/Qwen2.5-72B-Instruct")
TEMPERATURE = float(llm_config.get("temperature", 0.5))  # 默认温度

# ========== 工具函数（is_valid_sentence、split_sentences、extract_sentences_from_json）保持不变 ==========
def is_valid_sentence(sentence, lang="zh"):
    if not sentence or len(sentence) < 10:
        return False
    if lang == "zh":
        if re.search(r'[。！？!?]("|”)?$', sentence):
            return True
        if re.search(r'["“”].+[。！？!?]["“”]$', sentence):
            return True
    else:
        if sentence[-1] in '.!?':
            return True
    return False

def split_sentences(text, lang="zh"):
    text = re.sub(r"\s+", " ", text)
    if lang == "zh":
        parts = re.split(r"(?<=[。！？!?])", text)
        sentences = []
        buffer = ""
        for part in parts:
            buffer += part
            if part.endswith(("。", "！", "？", "!", "?")):
                sentences.append(buffer)
                buffer = ""
        if buffer:
            sentences.append(buffer)
    else:
        sentences = re.split(r'(?<=[.!?])\s+', text)
    return [
        s.strip() for s in sentences
        if isinstance(s, str) and s.strip() and is_valid_sentence(s.strip(), lang)
    ]

def extract_sentences_from_json(json_path, discarded_log):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = []
    for page in data.get("pdf_info", []):
        page_num = page.get("page_idx", 0) + 1
        for block in page.get("para_blocks", []):
            if block.get("type") == "text":
                lines = ["".join(span.get("content", "") for span in line.get("spans", [])) for line in block.get("lines", [])]
                paragraph = " ".join(lines).strip()
                if not paragraph:
                    continue
                is_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in paragraph)
                lang = "zh" if is_chinese else "en"
                sentences = split_sentences(paragraph, lang=lang)
                if not sentences:
                    discarded_log.append({
                        "reason": "no valid sentence",
                        "text": paragraph,
                        "source": os.path.basename(json_path),
                        "page": page_num
                    })
                    continue
                for sent in sentences:
                    results.append({
                        "text": sent,
                        "page": page_num,
                        "bbox": block.get("bbox"),
                        "source": os.path.basename(json_path)
                    })
    return results

# ========== 关键修改：仅处理SELECTED_JSON_FILES中的文件 ==========
print("正在提取段落并拆分词条 ...")
all_paras = []
ignored_paragraphs = []

# 替换原有的“遍历JSON_DIR”逻辑，改为处理选中的文件
for file in SELECTED_JSON_FILES:
    # 1. 清理文件名（避免空格/特殊字符问题）
    file = file.strip()
    # 2. 检查文件格式（必须是JSON）
    if not file.endswith(".json"):
        print(f"⚠️ 跳过非JSON文件：{file}")
        continue
    # 3. 拼接完整路径
    json_path = os.path.join(JSON_DIR, file)
    # 4. 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"⚠️ 文件不存在，跳过：{json_path}")
        continue
    # 5. 提取段落（复用原有逻辑）
    print(f"🔍 正在处理：{file}")
    all_paras.extend(extract_sentences_from_json(json_path, ignored_paragraphs))

# ========== 新增：提取所有text并保存到JSON ==========
if all_paras:  # 仅当有有效文本时才保存
    # 方案2：保存文本及元数据
    all_texts = [
        {
            "text": para["text"],
            "source": para["source"],
            "page": para["page"]
        } 
        for para in all_paras
    ]

    # 写入JSON文件（放在循环外，确保只写一次）
    with open(ALL_TEXTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)
    print(f"所有有效文本已保存至: {ALL_TEXTS_JSON}")
else:
    print("没有提取到有效文本，不生成JSON文件")


# ========== 输出跳过的段落 ==========
if ignored_paragraphs:
    with open(BAD_PARAGRAPH_LOG, "w", encoding="utf-8") as f:
        for item in ignored_paragraphs:
            path = item.get("source", "unknown")
            page = item.get("page", "-")
            para = item.get("text", "")
            f.write(f"{path} - Page {page}\n{para}\n\n")
    print(f"已跳过无效段落 {len(ignored_paragraphs)} 条，记录见：{BAD_PARAGRAPH_LOG}")

# ========== 向量化 ==========
texts = [p["text"] for p in all_paras]
if not texts:
    print("无有效文本，程序终止")
    exit()

print("正在执行聚类 ...")
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download("BAAI/bge-m3")
model = SentenceTransformer(model_dir, device="cuda")
embeddings = model.encode(texts, show_progress_bar=True)

# ========== KMeans 聚类 ==========
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0)
labels = kmeans.fit_predict(embeddings)

# ========== 聚类收集 ==========
cluster_texts = defaultdict(list)
cluster_sources = defaultdict(set)
for i, label in enumerate(labels):
    cluster_texts[label].append(texts[i])
    cluster_sources[label].add(all_paras[i]["source"])

# ========== 使用 LLM 命名 ==========
def generate_cluster_label(label, paras, sources, client, LLM_MODEL, TEMPERATURE):
    """生成聚类类名（含重试机制，兼容样本不足5条的情况）"""
    max_retries = 5  # 最大重试次数
    retry_delay = 1  # 重试间隔（秒）
    max_samples = 5  # 最大样本数量
    
    for retry in range(max_retries):
        try:
            # 1. 准备样本数据（动态适配实际数量，最多5条）
            actual_samples = min(max_samples, len(paras))  # 取实际数量和最大数量的较小值
            sample_texts = paras[:actual_samples]  # 根据实际数量截取样本
            sources_str = ", ".join(list(sources)[:3])  # 取前3个来源
            
            # 2. 构建动态提示词（根据实际样本数量调整描述）
            prompt = f"""你是中文文本主题命名专家。请根据以下{actual_samples}条文本，生成一个精准且独特的类名：
要求：
- 不超过6个汉字
- 风格学术、中性，避免口语/网络词
- 精准反映文本核心主题，概括所有文本的共同点
- 保留与其他类别的区分度
- 直接输出类名，不加编号、解释和额外说明

文本样本（来源：{sources_str}）：
"""
            # 动态添加样本文本（数量根据实际情况变化）
            for i, text in enumerate(sample_texts, start=1):
                prompt += f"{i}. {text}\n"  # 每条文本单独编号
            
            # 3. 调用LLM生成类名
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,  # 降低温度提高稳定性（原1.0容易产生不稳定结果）
                extra_body={"enable_thinking": False}
            )
            label_name = response.choices[0].message.content.strip()
            
            # 4. 校验最终类名有效性
            if not label_name:
                raise ValueError("生成的类名为空")
            if not any('\u4e00' <= c <= '\u9fff' for c in label_name):  # 确保包含中文
                raise ValueError(f"类名不含有效中文字符：{label_name}")

            print(f"聚类 {label} 命名成功（尝试{retry+1}/{max_retries}）：{label_name}")
            return label_name

        except Exception as e:
            # 最后一次重试失败则使用备用命名
            if retry == max_retries - 1:
                print(f"聚类 {label} 命名失败（已达最大重试次数）：{str(e)}")
                # 生成备用名称（基于文本哈希）
                import hashlib
                text_hash = hashlib.md5("".join(paras[:3]).encode()).hexdigest()[:4]
                return f"主题_{text_hash}"
            # 否则打印警告并重试
            print(f"聚类 {label} 命名尝试{retry+1}/{max_retries}失败：{str(e)}，将重试...")
            import time
            time.sleep(retry_delay * (retry + 1))  # 指数退避延迟

    # 最终 fallback（理论上不会触发）
    import hashlib
    text_hash = hashlib.md5("".join(paras[:3]).encode()).hexdigest()[:4]
    return f"主题_{text_hash}"


# 主循环调用（替换原for循环）
cluster_labels = {}
for label, paras in cluster_texts.items():
    # 获取该聚类的来源信息
    cluster_sources_list = cluster_sources.get(label, set())
    # 调用带重试机制的命名函数
    cluster_labels[label] = generate_cluster_label(
        label=label,
        paras=paras,
        sources=cluster_sources_list,
        client=client,
        LLM_MODEL=LLM_MODEL,
        TEMPERATURE=TEMPERATURE
    )

# ========== 输出最终文件 ==========
print("正在写入结果文件 ...")

# --- 1. 构建结构化数据 (用于 JSON 和 CSV) ---
output_data = {}
# 为 CSV 准备数据，存储所有段落及其聚类信息
clustered_csv_data = []

# 对聚类标签进行排序，确保一致性
sorted_labels = sorted(cluster_labels.keys())

# 新增：用于记录已使用的带编号类别名，避免重复
used_numbered_names = set()

for i, label in enumerate(sorted_labels, start=1):
    original_name = cluster_labels[label]
    # 新增：处理名称重复逻辑，若 original_name 对应的 numbered_name 已存在，则追加后缀区分
    base_numbered_name = f"{i}.{original_name}"
    numbered_name = base_numbered_name
    count = 1
    while numbered_name in used_numbered_names:
        numbered_name = f"{i}.{original_name}_{count}"
        count += 1
    used_numbered_names.add(numbered_name)
    cluster_labels[label] = numbered_name   # 更新 cluster_labels 字典

    para_indices = [j for j, l in enumerate(labels) if l == label]

    # 初始化 JSON 输出的列表
    output_data[numbered_name] = []

    for j in para_indices:
        para = all_paras[j]
        para_dict = {
            "text": para.get("text", ""),
            "source": para.get("source", "未知来源"),
            "page": para.get("page", -1),
            "root_class": numbered_name # 使用带编号的类名
        }
        # 添加到 JSON 结构
        output_data[numbered_name].append(para_dict)

        # 为聚类结果 CSV 准备数据
        csv_row = {
            "cluster_id": label,           # 原始聚类ID (数字)
            "root_class": numbered_name,   # 带编号和名称的聚类标签
            "text": para_dict["text"],
            "source": para_dict["source"],
            "page": para_dict["page"]
        }
        clustered_csv_data.append(csv_row)


# --- 2. 写入 JSON 文件 (保持原样) ---
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)
print(f"聚类结果已保存至 (JSON): {OUTPUT_FILE}")

# --- 3. 写入聚类结果 CSV 文件 ---
# import csv # 引入 csv 模块
# CLUSTER_CSV_FILE = OUTPUT_FILE.replace(".json", ".csv") # 生成 CSV 文件名
# try:
#     if clustered_csv_data: # 确保有数据再写入
#         with open(CLUSTER_CSV_FILE, "w", newline="", encoding="utf-8") as csvfile:
#             fieldnames = ["cluster_id", "root_class", "text", "source", "page"]
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#             writer.writeheader()
#             writer.writerows(clustered_csv_data)
#         print(f"聚类结果已保存至 (CSV): {CLUSTER_CSV_FILE}")
#     else:
#         print("警告: 没有聚类数据可写入 CSV 文件。")
# except Exception as e:
#     print(f"写入聚类结果 CSV 文件时出错: {e}")
# --- 3. 写入聚类结果 Excel 文件 ---
import pandas as pd

CLUSTER_EXCEL_FILE = OUTPUT_FILE.replace(".json", ".xlsx")  # 生成 Excel 文件名

try:
    if clustered_csv_data:  # 确保有数据再写入
        df = pd.DataFrame(clustered_csv_data)
        df.to_excel(CLUSTER_EXCEL_FILE, index=False, engine='openpyxl')
        print(f"聚类结果已保存至 (Excel): {CLUSTER_EXCEL_FILE}")
    else:
        print("警告: 没有聚类数据可写入 Excel 文件。")
except Exception as e:
    print(f"写入聚类结果 Excel 文件时出错: {e}")

# --- 4. 写入被忽略段落的日志文件 ---
# TXT 文件 (保持原样)
if ignored_paragraphs:
    with open(BAD_PARAGRAPH_LOG, "w", encoding="utf-8") as f:
        for item in ignored_paragraphs:
            path = item.get("source", "unknown")
            page = item.get("page", "-")
            para = item.get("text", "")
            f.write(f"{path} - Page {page}\n{para}\n\n")
    print(f"已跳过无效段落 {len(ignored_paragraphs)} 条，记录见 (TXT): {BAD_PARAGRAPH_LOG}")
else:
    print("没有段落被跳过。")

# CSV 文件 (新增)
# IGNORED_CSV_FILE = BAD_PARAGRAPH_LOG.replace(".txt", ".csv") # 生成 CSV 文件名
# try:
#     if ignored_paragraphs: # 确保有数据再写入
#         with open(IGNORED_CSV_FILE, "w", newline="", encoding="utf-8") as csvfile:
#             # 定义 CSV 列名，通常与 ignored_paragraphs 字典的键对应
#             fieldnames = ["source", "page", "reason", "text"]
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#             writer.writeheader()
#             # 遍历 ignored_paragraphs 列表，写入每一行
#             for item in ignored_paragraphs:
#                 # 确保字段名与 fieldnames 匹配
#                 csv_row = {
#                     "source": item.get("source", "unknown"),
#                     "page": item.get("page", "-"),
#                     "reason": item.get("reason", "unknown"), # 确保 extract_sentences_from_json 中有记录 reason
#                     "text": item.get("text", "")
#                 }
#                 writer.writerow(csv_row)
#         print(f"已跳过无效段落记录已保存至 (CSV): {IGNORED_CSV_FILE}")
#     else:
#         print("没有段落被跳过，无需生成 CSV 日志。")
# except Exception as e:
#     print(f"写入跳过段落 CSV 文件时出错: {e}")

# --- 写入被忽略段落的日志文件（Excel 版）---
IGNORED_EXCEL_FILE = BAD_PARAGRAPH_LOG.replace(".txt", ".xlsx")  # 生成 Excel 文件名

try:
    if ignored_paragraphs:  # 确保有数据再写入
        # 构造 DataFrame
        data = [
            {
                "source": item.get("source", "unknown"),
                "page": item.get("page", "-"),
                "reason": item.get("reason", "unknown"),
                "text": item.get("text", "")
            }
            for item in ignored_paragraphs
        ]
        df = pd.DataFrame(data)

        # 写入 Excel
        df.to_excel(IGNORED_EXCEL_FILE, index=False, engine='openpyxl')
        print(f"已跳过无效段落记录已保存至 (Excel): {IGNORED_EXCEL_FILE}")
    else:
        print("没有段落被跳过，无需生成 Excel 日志。")
except Exception as e:
    print(f"写入跳过段落 Excel 文件时出错: {e}")

print("所有文件输出完成。")


