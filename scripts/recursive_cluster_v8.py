import json
import os
import re
import jieba
import hashlib
import numpy as np
from typing import List, Dict, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from filter_sentence import is_valid_sentence, load_stopwords, clean_text
import argparse 

MAX_RECURSION_DEPTH = 30

MAX_KEYWORDS = 3
MAX_CLASS_NAME_CHARS = 15

RANDOM_STATE = 42
STOPWORDS_PATH = 'cn_hit.txt'  # 中文停用词路径（可修改）


def is_valid_sentence(text: str) -> bool:
    """判断词条是否有效 (示例实现)"""
    return isinstance(text, str) and len(text.strip()) > 5

def load_stopwords(filepath: str) -> Set[str]:
    """加载停用词表 (示例实现)"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # 假设停用词文件每行一个词
            stopwords = set(line.strip() for line in f if line.strip())
        # 可以在这里添加一些常见的英文停用词
        # common_english_stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "can"}
        # stopwords.update(common_english_stopwords)
        return stopwords
    except FileNotFoundError:
        print(f"[警告] 停用词文件未找到: {filepath}，将使用空停用词表。")
        return set()

def clean_text(text: str) -> str:
    """清洗文本 (示例实现 - 保留中英文和基本标点用于分词)"""
    if not isinstance(text, str):
        return ""
    # 保留中文、英文单词、数字和基本标点（用于分词）
    # 可根据需要调整正则表达式
    cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\-_\.!?;:,]', ' ', text)
    # 将多个空格替换为一个
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

# ==== 停用词加载 ====
try:
    stopwords = load_stopwords(STOPWORDS_PATH)
except Exception as e:
    print(f"[错误] 加载停用词失败: {e}")
    stopwords = set()

# ==== 改进的混合语言分词器 ====
def mixed_tokenizer(text: str) -> List[str]:
    """
    支持中英文的分词器。
    """
    if not isinstance(text, str):
        return []
    tokens = []
    
    # --- 处理中文 ---
    chinese_parts = re.findall(r'[\u4e00-\u9fa5]+', text)
    for part in chinese_parts:
        # 使用 jieba 分词
        chinese_tokens = jieba.cut(part)
        # 过滤停用词和长度小于2的词
        filtered_chinese = [word for word in chinese_tokens if word not in stopwords and len(word) > 1]
        tokens.extend(filtered_chinese)

    # --- 处理英文 ---
    # 使用正则表达式提取英文单词 (考虑撇号，如 don't)
    english_parts = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)
    for part in english_parts:
        word = part.lower() # 转为小写
        # 过滤停用词和长度小于2的词
        if word not in stopwords and len(word) > 1:
            tokens.append(word)
            
    # --- 可选：处理数字 ---
    # number_parts = re.findall(r'\d+', text)
    # tokens.extend(number_parts) # 或者根据需要处理数字

    return tokens

def chinese_tokenizer(text):
    return [word for word in jieba.cut(text) if word not in stopwords and len(word) > 1]


# ==== 类名生成 ====
def generate_class_name(texts: List[str], method: str = 'tfidf', parent_name: str = "") -> str:
    texts = [t for t in texts if is_valid_sentence(t)]
    fallback_base = "子主题"
    if not texts:
        return fallback_base
    try:
        cleaned_texts = [clean_text(t) for t in texts]
        # 过滤掉清洗后为空的文本
        cleaned_texts = [t for t in cleaned_texts if t.strip()]
        if not cleaned_texts:
             raise ValueError("清洗后无有效文本")

        if method == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=1000, tokenizer=mixed_tokenizer)
            X = vectorizer.fit_transform(cleaned_texts)
            indices = np.argsort(np.asarray(X.sum(axis=0)).ravel())[::-1]

            # 只保留中文关键词，最多取前 MAX_KEYWORDS 个
            keywords = [
                vectorizer.get_feature_names_out()[i]
                for i in indices
                if re.search(r'[\u4e00-\u9fa5]', vectorizer.get_feature_names_out()[i])
            ][:MAX_KEYWORDS]

            if not keywords or len(keywords) < 1:
                raise ValueError("无有效中文关键词")

            class_segment = "/".join(keywords)

        elif method == 'center':
            vectorizer = TfidfVectorizer(tokenizer=mixed_tokenizer, max_features=1000)
            X = vectorizer.fit_transform(cleaned_texts)
            if X.shape[1] == 0:
                 raise ValueError("TF-IDF 向量化后无特征")
            sim_matrix = cosine_similarity(X)
            scores = sim_matrix.sum(axis=1)
            center_text = cleaned_texts[np.argmax(scores)]
            class_segment = truncate_text(center_text, MAX_CLASS_NAME_CHARS)
        elif method == 'gen':
            raise NotImplementedError("尚未实现生成式类名")
        else:
            raise ValueError("未知命名方法")
            
        # --- 确保类名长度符合要求 ---
        return truncate_text(class_segment, MAX_CLASS_NAME_CHARS)

    except Exception as e:
        # print(f"[调试] generate_class_name 失败: {e}") # 可选调试信息
        hash_suffix = hashlib.md5(" ".join(texts).encode()).hexdigest()[:4]
        return f"{fallback_base}{hash_suffix}"

def truncate_text(text: str, max_len: int) -> str:
    """
    截断文本，只保留中文字符和 '/'，并限制最大长度。
    """
    if not isinstance(text, str):
        return ""
    # 只保留中文字符和 '/'  (假设 '/' 是您用来分隔关键词的特定字符)
    cleaned_text = re.sub(r'[^\u4e00-\u9fa5/]', '', text) 
    # 截断到指定长度
    truncated = cleaned_text
    return truncated.strip()

# ==== 聚类逻辑 ====
def cluster_texts(texts: List[str], max_per_cluster: int = 20) -> Dict[str, List[int]]:
    if len(texts) <= max_per_cluster:
        return {'all': list(range(len(texts)))}
    
    cleaned_texts = [clean_text(t) for t in texts]
    # 过滤掉清洗后为空的文本
    cleaned_texts = [t for t in cleaned_texts if t.strip()]
    
    # 新增判断：如果清洗后全是空的，就不聚类
    if not cleaned_texts or all(t == '' for t in cleaned_texts):
        return {'all': list(range(len(texts)))}
        
    try:
        k = max(2, len(texts) // max_per_cluster)
        # 添加 max_features 限制向量维度
        vectorizer = TfidfVectorizer(tokenizer=mixed_tokenizer, max_features=5000) 
        X = vectorizer.fit_transform(cleaned_texts)
        if X.shape[1] == 0: # 如果没有特征
            return {'all': list(range(len(texts)))}
            
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10) # 增加 n_init 以提高稳定性
        labels = kmeans.fit_predict(X)
        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(str(label), []).append(idx)
        return clusters
    except ValueError as ve:
        print(f"[调试] KMeans ValueError: {ve}") # 可选调试信息
        return {'all': list(range(len(texts)))}
    except Exception as e:
        print(f"[调试] cluster_texts 失败: {e}") # 可选调试信息
        return {'all': list(range(len(texts)))}

def recursive_cluster_numbered(
    data_list: List[Dict],
    root_name: str,
    prefix_path: str,
    full_key_path: str,
    depth: int = 1,
    filtered_log=None
) -> Dict:
    if depth > MAX_RECURSION_DEPTH:
        return {f"{prefix_path}.{root_name}_深度超限": data_list}
        
    valid_items = []
    valid_indices = []
    for idx, item in enumerate(data_list):
        text = item.get("text", "")
        if is_valid_sentence(text):
            valid_items.append(item)
            valid_indices.append(idx)
        else:
            if filtered_log is not None:
                filtered_log.append(item)
                
    if not valid_items:
        return {f"{prefix_path}.{root_name}_无有效文本": data_list}
        
    texts = [item["text"] for item in valid_items]
    clusters = cluster_texts(texts, MAX_CLUSTER_SIZE)
    result = {}
    name_set = set()
    count = 1
    
    for label, local_indices in clusters.items():
        global_indices = [valid_indices[i] for i in local_indices]
        sub_items = [data_list[i] for i in global_indices]
        sub_texts = [item["text"] for item in sub_items]
        
        cluster_name = generate_class_name(sub_texts, method=CLASS_NAME_METHOD, parent_name="")
        orig_name = cluster_name
        suffix = 1
        while cluster_name in name_set:
            cluster_name = f"{orig_name}_{suffix}"
            suffix += 1
        name_set.add(cluster_name)
        
        new_prefix = f"{prefix_path}-{count}" if prefix_path else str(count)
        key = f"{new_prefix}.{root_name}_{cluster_name}"
        
        # 安全限制 key 长度
        if len(key) > 250:
            key = key[:250] + "_trunc"
            
        # --- 调整递归终止条件 ---
        # 基本停止条件：簇大小足够小
        if len(sub_items) <= MAX_CLUSTER_SIZE:
            result[key] = sub_items
        # 特殊停止条件：如果聚类结果不佳（例如所有点在一个簇），避免无限递归
        elif len(clusters) <= 1:
             # 如果只有一个簇，但数据量远大于 MAX_CLUSTER_SIZE，则强制再分一次（需要调整 k）
             # 或者简单地作为叶子节点停止，避免死循环
             # 这里选择作为叶子节点停止
             result[key] = sub_items
        else:
            # 正常递归
            sub_result = recursive_cluster_numbered(
                sub_items,
                root_name=f"{root_name}_{cluster_name}",
                prefix_path=new_prefix,
                full_key_path=key,
                depth=depth + 1,
                filtered_log=filtered_log
            )
            # 如果递归返回的是一个字典（包含子簇），则展开；否则直接赋值
            # 注意：这里的逻辑取决于 recursive_cluster_numbered 的返回结构
            # 当前实现中，它总是返回一个字典，所以直接赋值即可
            result[key] = sub_result 
        count += 1
    return result

# ==== 输出结构 ====
def print_class_tree(tree: Dict, indent: str = "", level: int = 0):
    for class_name, content in tree.items():
        prefix = "📂" if isinstance(content, dict) else "├──"
        print(f"{indent}{prefix} {class_name}")
        if isinstance(content, dict):
            print_class_tree(content, indent + "  ", level + 1)

def build_class_tree_dict(tree: Dict) -> Dict:
    result = {}
    for class_name, content in tree.items():
        if isinstance(content, dict):
            result[class_name] = build_class_tree_dict(content)
        else:
            # 如果 content 是列表（叶子节点），可以返回 None 或其他标识
            result[class_name] = None 
    return result


from sklearn.cluster import AgglomerativeClustering
def hierarchical_cluster_numbered(
    data_list: List[Dict],
    root_name: str,
    prefix_path: str,
    full_key_path: str,
    max_cluster_size: int = 20,
    filtered_log=None
) -> Dict:
    """
    使用凝聚式层次聚类（Agglomerative Clustering）对文本进行分层聚类。
    """
    valid_items = []
    valid_indices = []
    for idx, item in enumerate(data_list):
        text = item.get("text", "")
        if is_valid_sentence(text):
            valid_items.append(item)
            valid_indices.append(idx)
        else:
            if filtered_log is not None:
                filtered_log.append(item)

    if not valid_items:
        return {f"{prefix_path}.{root_name}_无有效文本": data_list}

    texts = [item["text"] for item in valid_items]
    n_samples = len(texts)

    if n_samples <= max_cluster_size:
        return {f"{prefix_path}.{root_name}_单簇": data_list}

    # 清洗并向量化
    cleaned_texts = [clean_text(t) for t in texts]
    cleaned_texts = [t for t in cleaned_texts if t.strip()]
    if not cleaned_texts or all(t == '' for t in cleaned_texts):
        return {f"{prefix_path}.{root_name}_无特征": data_list}

    try:
        vectorizer = TfidfVectorizer(tokenizer=mixed_tokenizer, max_features=5000)
        X = vectorizer.fit_transform(cleaned_texts)
        if X.shape[1] == 0:
            return {f"{prefix_path}.{root_name}_无向量": data_list}

        # 计算最大聚类数：至少每个簇不超过 max_cluster_size
        max_k = max(2, n_samples // max_cluster_size)
        n_clusters = max_k

        # 使用凝聚式聚类
        clustering_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average',
            compute_distances=True  # 便于可视化（可选）
        )
        # 注意：AgglomerativeClustering 不支持稀疏矩阵，需转为 dense
        X_dense = X.toarray()
        labels = clustering_model.fit_predict(X_dense)

        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(str(label), []).append(idx)

        # 生成聚类树结构（与 recursive 结构一致）
        result = {}
        name_set = set()
        count = 1

        for label, local_indices in clusters.items():
            global_indices = [valid_indices[i] for i in local_indices]
            sub_items = [data_list[i] for i in global_indices]
            sub_texts = [item["text"] for item in sub_items]

            cluster_name = generate_class_name(sub_texts, method=CLASS_NAME_METHOD, parent_name="")
            orig_name = cluster_name
            suffix = 1
            while cluster_name in name_set:
                cluster_name = f"{orig_name}_{suffix}"
                suffix += 1
            name_set.add(cluster_name)

            new_prefix = f"{prefix_path}-{count}" if prefix_path else str(count)
            key = f"{new_prefix}.{root_name}_{cluster_name}"

            # 限制 key 长度
            if len(key) > 250:
                key = key[:250] + "_trunc"

            # 如果子簇足够小，作为叶子节点；否则继续聚类（可选：递归层次聚类）
            # 如果你想做完整树状层次聚类，可递归调用 hierarchical_cluster_numbered
            if len(sub_items) <= max_cluster_size:
                result[key] = sub_items
            else:
                # 可选：递归层次聚类 → 构建更深的树
                sub_result = hierarchical_cluster_numbered(
                    sub_items,
                    root_name=f"{root_name}_{cluster_name}",
                    prefix_path=new_prefix,
                    full_key_path=key,
                    max_cluster_size=max_cluster_size,
                    filtered_log=filtered_log
                )
                result[key] = sub_result
            count += 1

        return result

    except Exception as e:
        print(f"[调试] 层次聚类失败: {e}")
        return {f"{prefix_path}.{root_name}_聚类失败": data_list}

# ==== 主程序 ====
def process_json_file(input_path: str, output_path: str, clustering_mode: str = "recursive"):
    filtered_log = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"[错误] 输入文件未找到: {input_path}")
        return
    except json.JSONDecodeError as e:
        print(f"[错误] JSON 解码失败: {input_path}, {e}")
        return

    final_data = {}
    for orig_label, items in raw_data.items():
        if "." in orig_label:
            prefix_num, root_name = orig_label.split(".", 1)
        else:
            prefix_num, root_name = "1", orig_label

        print(f"[处理] 开始聚类大类: {orig_label}")

        if clustering_mode == "recursive":
            cluster_result = recursive_cluster_numbered(
                items,
                root_name=root_name,
                prefix_path=prefix_num,
                full_key_path="",
                filtered_log=filtered_log
            )
        elif clustering_mode == "hierarchical":
            cluster_result = hierarchical_cluster_numbered(
                items,
                root_name=root_name,
                prefix_path=prefix_num,
                full_key_path="",
                max_cluster_size=MAX_CLUSTER_SIZE,
                filtered_log=filtered_log
            )
        else:
            print(f"[错误] 不支持的聚类模式: {clustering_mode}")
            return

        final_data[orig_label] = cluster_result

    print("\n=== 类名聚类结构树 ===")
    print_class_tree(final_data)

    # 保存结果（略）
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        print(f"[完成] 已保存聚类结果到: {output_path}")
    except Exception as e:
        print(f"[错误] 保存聚类结果失败: {output_path}, {e}")

    tree_dict = build_class_tree_dict(final_data)
    tree_path = os.path.splitext(output_path)[0] + "_tree.json"
    try:
        with open(tree_path, 'w', encoding='utf-8') as f:
            json.dump(tree_dict, f, indent=2, ensure_ascii=False)
        print(f"[完成] 已保存聚类树结构到: {tree_path}")
    except Exception as e:
        print(f"[错误] 保存聚类树结构失败: {tree_path}, {e}")

    if filtered_log:
        filtered_path = os.path.splitext(output_path)[0] + "_filtered_out.json"
        try:
            with open(filtered_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_log, f, ensure_ascii=False, indent=2)
            print(f"[提示] 已保存被过滤掉的无效文本：{filtered_path}")
        except Exception as e:
            print(f"[错误] 保存过滤日志失败: {filtered_path}, {e}")


# ==== 执行 ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="递归或层次聚类脚本")
    parser.add_argument("--input", required=True, help="输入JSON文件路径")
    parser.add_argument("--output", required=True, help="输出JSON文件路径")
    parser.add_argument("--method", default="tfidf", choices=["tfidf", "center", "gen"], help="类名生成方法")
    parser.add_argument("--max_cluster_size", type=int, default=20, help="每个聚类最大词条数（默认: 20）")
    parser.add_argument("--clustering-mode", default="recursive", choices=["recursive", "hierarchical"], 
                        help="聚类模式：recursive（递归KMeans）或 hierarchical（凝聚式层次聚类）")

    args = parser.parse_args()

    # 设置全局变量
    global CLASS_NAME_METHOD, MAX_CLUSTER_SIZE
    CLASS_NAME_METHOD = args.method
    MAX_CLUSTER_SIZE = args.max_cluster_size

    process_json_file(args.input, args.output, clustering_mode=args.clustering_mode)