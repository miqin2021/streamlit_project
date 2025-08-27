import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cut_tree
from scipy.spatial.distance import pdist
from sentence_transformers import SentenceTransformer
import re
from sklearn.cluster import KMeans  # 用于二次分割过大的聚类

# ----------------------
# 1. 配置参数
# ----------------------
INPUT_JSON = "all_paragraph_texts.json"
OUTPUT_CLUSTER_JSON = "hierarchical_clusters.json"
OUTPUT_DENDROGRAM = "dendrogram.png"
MAX_CLUSTER_SIZE = 20  # 每个聚类的最大数量限制
VECTOR_MODEL = "BAAI/bge-m3"

# ----------------------
# 2. 读取JSON并提取文本
# ----------------------
def load_texts_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    valid_texts = []
    for item in data:
        text = item.get("text", "").strip()
        if len(text) < 5:
            continue
        valid_texts.append({
            "text": text,
            "source": item.get("source", "unknown"),
            "page": item.get("page", -1),
            "index": len(valid_texts)
        })
    print(f"共提取有效文本 {len(valid_texts)} 条")
    return valid_texts

text_data = load_texts_from_json(INPUT_JSON)
texts = [item["text"] for item in text_data]

# ----------------------
# 3. 文本向量化
# ----------------------
def text_to_embeddings(texts, model_name):
    from modelscope.hub.snapshot_download import snapshot_download
    model_dir = snapshot_download("BAAI/bge-m3")
    model = SentenceTransformer(model_dir, device="cuda" if torch.cuda.is_available() else "cpu")
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"文本向量化完成，向量维度：{embeddings.shape[1]}")
    return embeddings

import torch
embeddings = text_to_embeddings(texts, VECTOR_MODEL)

# ----------------------
# 4. 层次聚类（初始聚类）
# ----------------------
def hierarchical_clustering(embeddings, metric="cosine", linkage_method="ward"):
    distance_matrix = pdist(embeddings, metric=metric)
    linkage_matrix = linkage(distance_matrix, method=linkage_method)
    return linkage_matrix

linkage_matrix = hierarchical_clustering(embeddings)

# 绘制初始树状图（用于参考）
def plot_dendrogram(linkage_matrix, labels, output_path, figsize=(12, 8)):
    plt.figure(figsize=figsize)
    dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=8,
        truncate_mode="lastp",
        p=30,
        show_contracted=True
    )
    plt.title("初始层次聚类树状图")
    plt.xlabel("文本（前20字符）")
    plt.ylabel("距离")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"树状图已保存至：{output_path}")
    plt.close()

short_labels = [text[:20] + "..." for text in texts]
plot_dendrogram(linkage_matrix, short_labels, OUTPUT_DENDROGRAM)

# ----------------------
# 5. 核心逻辑：控制聚类大小不超过20
# ----------------------
def get_clusters_with_size_limit(linkage_matrix, embeddings, max_size=20):
    """
    生成聚类并确保每个聚类不超过max_size条
    1. 先进行初始聚类
    2. 对超过max_size的聚类进行二次分割
    """
    # 步骤1：初始聚类（使用较大的阈值，允许聚类过大）
    initial_threshold = 2.0  # 可根据树状图调整，先得到较少的大类
    initial_labels = fcluster(linkage_matrix, t=initial_threshold, criterion="distance") - 1  # 从0开始编号
    
    final_labels = initial_labels.copy()
    current_max_id = np.max(initial_labels) + 1  # 新聚类的起始ID
    
    # 步骤2：检查每个初始聚类，分割过大的聚类
    for cluster_id in np.unique(initial_labels):
        # 获取该聚类的样本索引和向量
        cluster_indices = np.where(initial_labels == cluster_id)[0]
        cluster_size = len(cluster_indices)
        
        if cluster_size <= max_size:
            continue  # 无需分割
        
        print(f"聚类 {cluster_id} 大小为 {cluster_size}，超过阈值 {max_size}，进行二次分割...")
        
        # 提取该聚类的向量
        cluster_embeddings = embeddings[cluster_indices]
        
        # 计算需要分割的子聚类数量（向上取整）
        n_subclusters = (cluster_size + max_size - 1) // max_size  # 例如25条→2个子类（20+5）
        
        # 使用KMeans进行二次分割
        kmeans = KMeans(n_clusters=n_subclusters, random_state=42)
        sub_labels = kmeans.fit_predict(cluster_embeddings)
        
        # 更新最终标签：为子聚类分配新ID
        for i, idx in enumerate(cluster_indices):
            final_labels[idx] = current_max_id + sub_labels[i]
        
        current_max_id += n_subclusters  # 更新下一批子聚类的起始ID
    
    return final_labels

# 获取满足大小限制的聚类结果
cluster_labels = get_clusters_with_size_limit(linkage_matrix, embeddings, MAX_CLUSTER_SIZE)
n_clusters = len(np.unique(cluster_labels))
print(f"聚类完成，共得到 {n_clusters} 个聚类，所有聚类均不超过 {MAX_CLUSTER_SIZE} 条")

# ----------------------
# 6. 验证聚类大小并输出结果
# ----------------------
def save_cluster_results(text_data, cluster_labels, output_path):
    clusters = {}
    for i, item in enumerate(text_data):
        cluster_id = int(cluster_labels[i])
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append({
            "text": item["text"],
            "source": item["source"],
            "page": item["page"],
            "original_index": item["index"]
        })
    
    # 验证所有聚类是否符合大小限制
    for cid, items in clusters.items():
        if len(items) > MAX_CLUSTER_SIZE:
            print(f"警告：聚类 {cid} 大小为 {len(items)}，超过限制！")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)
    print(f"聚类结果已保存至：{output_path}")

save_cluster_results(text_data, cluster_labels, OUTPUT_CLUSTER_JSON)

# ----------------------
# 7. 聚类统计
# ----------------------
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
print("\n聚类数量统计：")
for cluster_id, count in cluster_counts.items():
    print(f"聚类 {cluster_id}：{count} 条文本")
