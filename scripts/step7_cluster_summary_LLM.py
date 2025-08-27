import os
import json
import re
import argparse
import time
import torch
import numpy as np
from collections import defaultdict, UserDict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from openai import OpenAI
import pandas as pd
from modelscope.hub.snapshot_download import snapshot_download as ms_download

# 解析命令行参数（核心变更：支持指定聚类数量区间）
def parse_args():
    parser = argparse.ArgumentParser(description="对根类别下的 summary 进行二次聚类和总结")
    parser.add_argument("--input", required=True, help="Step6 生成的 summary JSON 文件")
    parser.add_argument("--output", required=True, help="Step7 输出结果文件路径")
    parser.add_argument("--min_size", type=int, required=True, 
                      help="每个聚类的最小词条数量（如5）")
    parser.add_argument("--max_size", type=int, required=True, 
                      help="每个聚类的最大词条数量（如10）")
    parser.add_argument("--prompt_file", required=True, help="LLM 提示词文件路径")
    return parser.parse_args()

args = parse_args()

# 校验参数合理性
if args.min_size <= 0 or args.max_size <= 0 or args.min_size >= args.max_size:
    raise ValueError("参数错误：min_size必须小于max_size，且均为正数")

# 配置路径
OUTPUT_DIR = os.path.dirname(args.output)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载 LLM 配置
CONFIG_FILE = "llm.config"
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    llm_config = json.load(f)

client = OpenAI(
    base_url=llm_config["base_url"],
    api_key=llm_config["api_key"]
)
LLM_MODEL = llm_config.get("model_name", "Qwen/Qwen2.5-72B-Instruct")  # 默认模型
TEMPERATURE = float(llm_config.get("temperature", 0.5))  # 默认温度

# 加载提示词
with open(args.prompt_file, "r", encoding="utf-8") as f:
    base_prompt = f.read()

# 预加载嵌入模型（全局只加载一次）
print("预加载句子嵌入模型...")
try:
    model_dir = ms_download("BAAI/bge-m3")
    model = SentenceTransformer(
        model_dir, 
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"模型加载成功，使用设备: {model.device}")
except Exception as e:
    print(f"模型加载失败，将使用默认模型: {e}")
    model = SentenceTransformer("BAAI/bge-m3")

# 加载 Step6 的 summary 数据
def load_step6_data(input_path):
    """从 Step6 JSON 提取根类别和对应的 summary"""
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"加载 Step6 数据失败: {e}")
        return defaultdict(list)
    
    grouped_data = defaultdict(list)
    
    for item in data:
        # 提取根类别
        root_class = "未知类别"
        class_str = item.get("class", "")
        
        if class_str:
            parts = class_str.split(".", 1)
            if len(parts) == 2:
                root_level = parts[0].split("-")[0]
                main_class = parts[1].split("_")[0]
                root_class = f"{root_level}.{main_class}"
        
        # 从句子中提取根类别作为备选
        if root_class == "未知类别" and item.get("sentences"):
            root_class = item["sentences"][0].get("root_class", "未知类别")
        
        # 添加有效的 summary
        summary = item.get("summary", "").strip()
        if summary:
            grouped_data[root_class].append(summary)
    
    return grouped_data

# 聚类核心函数（根据数量区间动态计算聚类数量）
def cluster_summaries(summaries, min_size, max_size, model):
    """
    对 summary 列表进行语义聚类
    - 支持指定每个聚类的词条数量区间 [min_size, max_size]
    - 自动计算合理的聚类数量并选择最优方案
    """
    total = len(summaries)
    # 情况1：总数量小于最小阈值，不聚类
    if total < min_size:
        print(f"样本数({total})小于最小聚类规模({min_size})，不进行聚类")
        return {}
    
    # 情况2：总数量在区间内，直接作为一个聚类
    if min_size <= total <= max_size:
        print(f"样本数({total})在聚类区间[{min_size},{max_size}]内，作为单一聚类")
        return {0: [{"summary": s, "index": i} for i, s in enumerate(summaries)]}
    
    # 情况3：总数量超过最大阈值，需要聚类
    # 计算可能的聚类数量范围
    min_clusters = max(1, (total + max_size - 1) // max_size)  # 向上取整
    max_clusters = max(1, total // min_size)
    
    # 确保聚类数量在合理范围
    if min_clusters > max_clusters:
        min_clusters = max_clusters
    
    print(f"根据区间[{min_size},{max_size}]，可能的聚类数量范围: {min_clusters}~{max_clusters}")
    
    # 生成嵌入
    try:
        print(f"生成 {total} 条 summary 的嵌入向量...")
        embeddings = model.encode(summaries, show_progress_bar=True)
    except Exception as e:
        print(f"生成嵌入失败: {e}")
        return {}
    
    # 寻找最优聚类数量（通过轮廓系数）
    best_k = min_clusters
    best_score = -1
    scores = []
    
    # 限制尝试的聚类数量（避免过多计算）
    k_candidates = range(min_clusters, min(max_clusters + 1, 20))  # 最多尝试20个值
    
    for k in k_candidates:
        try:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            # 计算轮廓系数（值越大聚类效果越好）
            score = silhouette_score(embeddings, labels)
            scores.append((k, score))
            if score > best_score:
                best_score = score
                best_k = k
        except Exception as e:
            print(f"尝试聚类数量 {k} 失败: {e}")
            continue
    
    print(f"最优聚类数量: {best_k} (轮廓系数: {best_score:.4f})")
    
    # 执行最优聚类
    try:
        kmeans = KMeans(n_clusters=best_k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(embeddings)
    except Exception as e:
        print(f"聚类失败: {e}")
        return {}
    
    # 分组结果
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append({
            "summary": summaries[idx],
            "index": idx
        })
    
    # 调整小聚类（合并过小的聚类到最相似的聚类）
    final_clusters = adjust_small_clusters(clusters, embeddings, min_size)
    return final_clusters

def adjust_small_clusters(clusters, embeddings, min_size):
    """合并小于最小规模的聚类到最相似的聚类"""
    # 计算每个聚类的中心
    cluster_centers = {}
    cluster_indices = {}
    for label, items in clusters.items():
        indices = [item["index"] for item in items]
        cluster_indices[label] = indices
        cluster_centers[label] = np.mean(embeddings[indices], axis=0)
    
    # 找出需要合并的小聚类
    small_clusters = [label for label, items in clusters.items() if len(items) < min_size]
    if not small_clusters:
        return clusters
    
    print(f"发现 {len(small_clusters)} 个小于最小规模的聚类，进行合并调整")
    
    # 合并小聚类到最相似的大聚类
    main_clusters = [label for label in clusters if label not in small_clusters]
    for small_label in small_clusters:
        # 找到最相似的主聚类
        small_center = cluster_centers[small_label]
        similarities = []
        for main_label in main_clusters:
            main_center = cluster_centers[main_label]
            # 计算余弦相似度
            sim = np.dot(small_center, main_center) / (np.linalg.norm(small_center) * np.linalg.norm(main_center))
            similarities.append((main_label, sim))
        
        # 合并到最相似的主聚类
        if similarities:
            best_main_label = max(similarities, key=lambda x: x[1])[0]
            clusters[best_main_label].extend(clusters[small_label])
            print(f"将小聚类 {small_label} 合并到聚类 {best_main_label}")
        # 删除已合并的小聚类
        del clusters[small_label]
    
    return clusters

# 生成总结（同时支持聚类和非聚类场景）
def generate_summary(group_id, summaries, root_class, prompt, is_cluster=True):
    """
    生成总结
    - is_cluster=True: 用于聚类后的分组总结
    - is_cluster=False: 用于未聚类的整体总结
    """
    max_retries = 5
    retry_delay = 1
    
    for retry in range(max_retries):
        try:
            # 准备样本
            sample_count = len(summaries)
            sample_texts = [s["summary"] if is_cluster else s for s in summaries]
            
            # 构建提示词
            scene = "聚类" if is_cluster else "整体"
            full_prompt = prompt.format_map(SafeDict({
                "summaries": "\n".join([f"{i+1}. {t}" for i, t in enumerate(sample_texts)]),
            }))
            
            # 调用 LLM
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=TEMPERATURE,
                extra_body={"enable_thinking": False}
            )
            
            summary = response.choices[0].message.content.strip()
            if not summary:
                raise ValueError(f"生成的{scene}总结为空")
            
            print(f"{scene}总结 {group_id} 生成成功（尝试 {retry+1}/{max_retries}）")
            return summary
            
        except Exception as e:
            if retry == max_retries - 1:
                print(f"{scene}总结 {group_id} 生成失败：{str(e)}，使用默认名称")
                import hashlib
                text_hash = hashlib.md5("".join([s["summary"] if is_cluster else s for s in summaries[:3]]).encode()).hexdigest()[:6]
                return f"{scene}总结_{group_id}_{text_hash}"
            
            print(f"{scene}总结 {group_id} 重试 {retry+1}/{max_retries}：{str(e)}")
            time.sleep(retry_delay * (retry + 1))
    
    return f"{scene}总结_{group_id}"

# 安全的提示词格式化工具
class SafeDict(UserDict):
    def __missing__(self, key):
        return f"{{{key}}}"  # 缺失键时返回原始格式

# 数据类型转换函数
def convert_to_python_types(obj):
    """递归转换所有numpy类型为Python原生类型"""
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# 主函数
def main():
    # 1. 加载并分组数据
    print("加载 Step6 的 summary 数据...")
    grouped_data = load_step6_data(args.input)
    print(f"共加载 {len(grouped_data)} 个根类别")
    
    if not grouped_data:
        print("没有有效数据，退出程序")
        return
    
    # 2. 处理每个根类别（新增：为根类别分配序号 root_idx）
    final_results = {}
    root_categories = list(grouped_data.keys())  # 固定根类别顺序
    for root_idx, root_class in enumerate(root_categories, start=1):
        summaries = grouped_data[root_class]
        total_summaries = len(summaries)
        print(f"\n处理根类别：{root_class}（{total_summaries} 条 summary，根类别序号：{root_idx}）")
        
        # 3. 执行聚类（根据区间自动决策）
        clusters = cluster_summaries(summaries, args.min_size, args.max_size, model)
        
        # 4. 生成总结（区分聚类和非聚类场景）
        if not clusters:
            # 不聚类，直接生成整体总结
            print(f"根类别 {root_class} 不满足聚类条件，生成整体总结")
            cluster_summaries_list = [{
                "cluster_id": 0,
                "summary": generate_summary(
                    group_id=0,
                    summaries=summaries,
                    root_class=root_class,
                    prompt=base_prompt,
                    is_cluster=False
                ),
                "original_summaries": summaries,
                "count": total_summaries,
                "numbered_id": f"T{root_idx}"  # 仅 T+根类别序号（无聚类时）
            }]
        else:
            # 为每个聚类生成总结
            cluster_summaries_list = []
            for cluster_id, cluster_items in clusters.items():
                summary_text = generate_summary(
                    group_id=cluster_id,
                    summaries=cluster_items,
                    root_class=root_class,
                    prompt=base_prompt,
                    is_cluster=True
                )
                
                # 生成 numbered_id：T{root_idx}-{cluster_id + 1}
                numbered_id = f"T{root_idx}-{cluster_id + 1}"  
                
                cluster_summaries_list.append({
                    "cluster_id": cluster_id,
                    "summary": summary_text,
                    "original_summaries": [item["summary"] for item in cluster_items],
                    "count": len(cluster_items),
                    "numbered_id": numbered_id
                })
            
            # 按 cluster_id 排序（保证编号顺序）
            cluster_summaries_list.sort(key=lambda x: x["cluster_id"])
        
        # 5. 保存结果
        final_results[root_class] = cluster_summaries_list
        print(f"根类别 {root_class} 处理完成，生成 {len(cluster_summaries_list)} 条总结")
    
    # 6. 保存 JSON 结果
    try:
        final_results = convert_to_python_types(final_results)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 结果保存至：{args.output}")
    except Exception as e:
        print(f"保存 JSON 失败：{e}")
        return
    
    # 7. 生成 Excel 版本
    excel_path = args.output.replace(".json", ".xlsx")
    try:
        excel_data = []
        for root_class, clusters in final_results.items():
            for cluster in clusters:
                excel_data.append({
                    "根类别": root_class,
                    "总结编号": cluster["numbered_id"],
                    "总结内容": cluster["summary"],
                    "包含原始总结数": cluster["count"],
                    "原始总结内容": "|".join(cluster["original_summaries"])
                })
        
        df = pd.DataFrame(excel_data)
        df.to_excel(excel_path, index=False, engine="openpyxl")
        print(f"Excel 结果保存至：{excel_path}")
    except Exception as e:
        print(f"生成 Excel 失败：{e}")

if __name__ == "__main__":
    main()

'''
python scripts/step7_cluster_summary_LLM.py \
    --input data/outputs/step6/step6_summary.json \
    --output data/outputs/step7/step7_summary-1126.json \
    --min_size 5 \
    --max_size 10 \
    --prompt_file data/outputs/step7/prompt_input.txt
'''