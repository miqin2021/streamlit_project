import json
from typing import List, Dict, Set

from openai import OpenAI
# ==== OpenAI / ModelScope 配置 ====
# 读取配置文件
with open("llm.config", "r", encoding="utf-8") as f:
    llm_config = json.load(f)

# 初始化客户端
client = OpenAI(
    base_url=llm_config["base_url"],
    api_key=llm_config["api_key"]
)
LLM_MODEL = llm_config.get("model_name", "Qwen/Qwen2.5-72B-Instruct")  # 默认模型
TEMPERATURE = float(llm_config.get("temperature", 0.5))  # 默认温度

from datetime import datetime
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

log("开始结合LLM进行总结...")
def summarize_texts_with_qwen(texts: List[str], client, custom_prompt: str = None) -> str:
    """
    用 Qwen 模型总结一组文本为一句话（含起因、经过、结果），只发送词条，不发送来源。
    可选自定义 prompt。
    """
    if not texts:
        return ""

    # 构造 prompt / messages
    if custom_prompt is None:
        content = "请对以下文本进行简洁总结：\n\n" + "\n".join(texts)
    else:
        content = custom_prompt + "\n\n" + "\n".join(texts)

    messages = [
        {"role": "user", "content": content}
    ]

    # # === ✅ 在发送前：保存传递给大模型的内容 ===
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # log_dir = "prompt_logs"
    # os.makedirs(log_dir, exist_ok=True)

    # log_file = os.path.join(log_dir, f"prompt_input_{timestamp}.json")

    # # 保存完整的输入请求结构
    # input_data = {
    #     "timestamp": timestamp,
    #     "model": "Qwen/Qwen2.5-72B-Instruct",
    #     "messages": messages,
    #     "temperature": 0.3
    # }

    # try:
    #     with open(log_file, "w", encoding="utf-8") as f:
    #         json.dump(input_data, f, ensure_ascii=False, indent=4)
    #     print(f"✅ 已保存发送给大模型的内容到: {os.path.abspath(log_file)}")
    # except Exception as e:
    #     print(f"❌ 保存 prompt 失败: {e}")
    # # === ✅ 在发送前：保存传递给大模型的内容 ===

    if custom_prompt:
        prompt = custom_prompt + "\n\n" + "\n".join(f"- {t}" for t in texts) + "\n\n请用一句话给出总结："
    else:
        prompt = (
            "以下是一些来自同一个类别但是可能不同语境的词条，请基于它们的整体内容，"
            "从客观、语言分析的角度出发，提炼一个简洁的一句话总结，"
            "可包含事件背景（起因）、主要过程（经过）、以及最后的变化或影响（结果）。"
            "无需评论立场，请仅描述现象与事实：\n\n"
            + "\n".join(f"- {t}" for t in texts)
            + "\n\n请用一句话给出总结："
        )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            extra_body={  
                "enable_thinking": False
            }
        )


        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[错误] Qwen 总结失败：{e}")
        return "（总结失败）"

def is_valid_sentence(text: str) -> bool:
    """判断词条是否有效 (示例实现)"""
    return isinstance(text, str) and len(text.strip()) > 5

def summarize_cluster_tree(tree: Dict, client, custom_prompt: str = None) -> Dict:

    """
    遍历聚类结果，为每个最底层子类添加 summary 字段，
    并记录每条词条的来源信息（含 source、page、root_class）。
    """
    summarized = {}
    for class_name, content in tree.items():
        if isinstance(content, dict):
            # 递归处理子树
            summarized[class_name] = summarize_cluster_tree(content, client, custom_prompt)

        else:
            # 叶子节点，处理文本和元数据 (假设 content 是列表)
            if isinstance(content, list):
                valid_items = [
                    item for item in content
                    if "text" in item and is_valid_sentence(item["text"])
                ]
                texts = [item["text"] for item in valid_items]
                # 记录每句话的来源信息
                sentences = [
                    {
                        "text": item["text"],
                        "source": item.get("source", "未知"),
                        "page": item.get("page", "未知"),
                        "root_class": item.get("root_class", "未知")
                    }
                    for item in valid_items
                ]
                # summary = summarize_texts_with_qwen(texts, client)
                summary = summarize_texts_with_qwen(texts, client, custom_prompt)
                summarized[class_name] = {
                    "summary": summary,
                    "count": len(texts),
                    "sentences": sentences
                }
            else:
                 # 如果 content 不是列表也不是字典，可能是 None 或其他，直接赋值
                 summarized[class_name] = content
    return summarized


def run_summary(input_path: str, output_path: str, custom_prompt: str = None):
    
    # Step 2: 总结（调用 Qwen）
    # 注意：需要确保 OpenAI client 已正确配置
    # 如果不需要总结功能，可以注释掉这部分
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)
        print("\n=== 开始生成总结 ===")
        summary_tree = summarize_cluster_tree(cluster_data, client, custom_prompt=custom_prompt)
        summary_path = output_path.replace(".json", "_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_tree, f, indent=2, ensure_ascii=False)
        print(f"[完成] 已保存总结结构到: {summary_path}")
    except Exception as e:
        print(f"[警告] 生成总结过程出错: {e}")

# ==== 执行 ====
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入JSON文件路径")
    parser.add_argument("--output", required=True, help="输出JSON文件路径")
    parser.add_argument("--prompt_file", type=str, default=None, help="自定义提示词文件路径")
    args = parser.parse_args()

    custom_prompt = None
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                custom_prompt = f.read().strip()
        except Exception as e:
            print(f"[警告] 读取 prompt 文件失败: {e}")
    run_summary(args.input, args.output, custom_prompt)