from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import os

def is_valid_sentence(text: str) -> bool:
    if not text:
        return False

    text = text.strip()
    if len(text) < 6:
        return False

    # 清除全角空格等
    text = text.replace("\u3000", "").replace("\xa0", "")

    text_lower = text.lower()

    # 关键词类过滤
    filter_keywords = [
        "关键词", "关键字", "作者", "基金", "来源", "编辑", "出处", "发布机构",
        "图", "表", "参考文献", "doi", "corresponding author", "about the author",
        "received", "accepted", "citation", "总结"
    ]
    if any(kw in text or kw in text_lower for kw in filter_keywords):
        return False

    # 图像、图表、图片引用
    if re.search(r'\[?image|图\s*\d+|fig(?:ure)?\s*\d+', text_lower):
        return False

    # 中英文字符比例
    chinese_chars = re.findall(r'[\u4e00-\u9fa5]', text)
    english_words = re.findall(r'[a-zA-Z]+', text)
    total_chars = len(text)
    chinese_ratio = len(chinese_chars) / total_chars
    english_ratio = len(" ".join(english_words)) / total_chars

    # 中文占比过低且无完整结构（如祈使句或片段）
    if chinese_ratio < 0.1 and len(english_words) < 5:
        return False

    # 判断是否语义完整（句末标点或谓语动词）
    if not re.search(r'[。！？.!?]$', text):
        # 若没有句尾标点，检查是否至少包含一个动词
        # 简化判断：中英文中是否出现常见动词
        common_verbs = ["是", "有", "进行", "认为", "成为", "包括", "加强", "推动"] + \
                       ["is", "are", "was", "were", "has", "have", "do", "does", "said", "show", "shows"]
        if not any(verb in text_lower for verb in common_verbs):
            return False

    return True

# ==== 停用词加载 ====
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def load_stopwords(file_path):
    """
    加载中英文停用词。
    """
    cn_words = set()
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                cn_words = set(line.strip() for line in f if line.strip())
        except Exception as e:
            print(f"[警告] 加载中文停用词文件 '{file_path}' 时出错: {e}")
            # 可以选择继续（只用英文停用词）或抛出异常

    else:
        print(f"[警告] 中文停用词文件未找到: {file_path}")
        # 可以选择使用默认停用词列表或继续（只用英文停用词）


    # 合并中文停用词和 sklearn 提供的英文停用词
    combined_stopwords = cn_words.union(ENGLISH_STOP_WORDS)
    return combined_stopwords


def clean_text(text: str) -> str:
    # 替换全角空格、特殊空白字符
    text = text.replace("\u3000", " ").replace("\xa0", " ").strip()

    # 去除无意义图片标记、URL等
    text = re.sub(r'\[image:.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+', '', text)

    # 保留中文、常见标点、数字、年份表达、单位（如“20世纪”、“98年”）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9.%/年月\-，。、“”‘’！!？?（）()\[\]：:；;]', '', text)

    # 替换英文双引号等为中文标点（可选）
    text = text.replace('"', '“').replace("'", "‘")

    # 清理多余空格和重复标点
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[，。]{2,}', '，', text)

    return text.strip()
