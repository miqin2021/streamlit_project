import os
import requests

# === 修改为本地 MinerU 的单一解析接口 ===
# PARSE_API = "http://localhost:8815/file_parse"
PARSE_API = "http://2ed1fa48.r20.cpolar.top/file_parse"
# -*- coding: utf-8 -*-
import os, json, re
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Union
import requests
import time
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mineru")

# === 修改为本地 MinerU 的单一解析接口 ===
# PARSE_API = "http://localhost:8815/file_parse"

INPUT_DIR = "./"
JSON_OUT_DIR = "data/0827-json_layout"
os.makedirs(JSON_OUT_DIR, exist_ok=True)

ALL_OUT_DIR = os.path.abspath("all_outdir")
os.makedirs(ALL_OUT_DIR, exist_ok=True)

MD_OUT_DIR = "all_output_md"
os.makedirs(MD_OUT_DIR, exist_ok=True)

# 支持的文档类型
SUPPORTED_EXTENSIONS = {'.pdf', '.doc', '.docx', '.xls', '.xlsx'}

def convert_to_pdf(input_path: Union[str, Path]) -> Path:
    """
    将办公文档转换为 PDF
    Args:
        input_path: 输入文件路径
    Returns:
        转换后的 PDF 路径
    Raises:
        RuntimeError: 转换失败
    """
    input_path = Path(input_path).resolve()
    
    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix not in ['.doc', '.docx', '.xls', '.xlsx']:
        logger.info(f"📎 文件无需转换: {input_path}")
        return input_path

    output_path = input_path.with_suffix('.pdf')

    cmd = [
        "libreoffice",
        "--headless",
        "--convert-to", "pdf",
        "--outdir", str(input_path.parent),
        str(input_path)
    ]

    try:
        logger.info(f"🔄 正在转换: {input_path} -> {output_path}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300  # 5分钟超时
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            raise RuntimeError(
                f"LibreOffice 转换失败 (code={result.returncode}):\n"
                f"命令: {' '.join(cmd)}\n"
                f"错误: {stderr}\n"
                f"输出: {stdout}"
            )

        if not output_path.exists():
            raise RuntimeError("转换完成但输出文件未生成")

        logger.info(f"✅ 转换成功: {input_path} -> {output_path}")
        return output_path

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"转换超时: {input_path}")
    except Exception as e:
        logger.error(f"❌ 文档转 PDF 失败: {e}")
        raise

def process_and_extract(pdf_path: str) -> dict:
    """
    解析单个文档（支持自动转换）
    Returns:
        {
            "file": str,
            "success": bool,
            "path": str or None,
            "error": str or None
        }
    """
    # ✅ 第一步：转换为 PDF（如果是 office 文档）
    pdf_path = convert_to_pdf(pdf_path)
    file_name = os.path.basename(pdf_path)
    original_name = os.path.splitext(file_name)[0]

    # 输出路径
    json_output_path = os.path.join(JSON_OUT_DIR, f"{original_name}_middle.json")
    data_md_output_path = os.path.join(MD_OUT_DIR, f"{original_name}.md")

    # 用于保存最终结果
    result = {}

    # 存放找到的 uuid_dir，用于 finally 删除
    uuid_dir_to_clean = None

    try:
        # ✅ 检查是否已有解析结果（跳过解析）
        try:
            flat_output_dir = os.path.join(ALL_OUT_DIR, original_name, "auto")
            middle_json_path = os.path.join(flat_output_dir, f"{original_name}_middle.json")
            if os.path.exists(middle_json_path):
                print(f"✅ 检测到已解析结果: {middle_json_path}")
                os.makedirs(JSON_OUT_DIR, exist_ok=True)
                shutil.copy2(middle_json_path, json_output_path)
                print(f"✅ 已复制到: {json_output_path}")
                return {
                    "file": original_name,
                    "success": True,
                    "path": json_output_path,
                    "error": None
                }

        except Exception as e:
            print(f"⚠️ 检查已存在解析结果时出错: {e}")

        # ✅ 检查本地是否有中间结果
        if os.path.exists(json_output_path):
            print(f"✅ 跳过已存在的文件: {json_output_path}")
            return {
                "file": original_name,
                "success": True,
                "path": json_output_path,
                "error": None
            }

        # === 主解析流程开始 ===
        with open(pdf_path, "rb") as f:
            files = [("files", (file_name, f, "application/pdf"))]
            data = {
                "language": "auto",
                "backend": "pipeline",
                "parse_method": "auto",
                "formula_enable": "true",
                "table_enable": "true",
                "return_md": "true",
                "return_middle_json": "true",
                "return_content_list": "false",
                "return_model_output": "false",
                "return_images": "false",
                "start_page_id": "0",
                "end_page_id": "-1",
                "output_dir": ALL_OUT_DIR
            }
            print(f"🚀 正在解析: {file_name}")
            response = requests.post(PARSE_API, files=files, data=data, timeout=300)

        if response.status_code != 200:
            result = {
                "file": original_name,
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
            return result

        try:
            resp_json = response.json()
        except Exception as e:
            result = {
                "file": original_name,
                "success": False,
                "error": f"响应不是合法 JSON: {e}"
            }
            return result

        results = resp_json.get("results")
        if not results:
            result = {
                "file": original_name,
                "success": False,
                "error": f"响应中无 'results' 字段: {list(resp_json.keys())}"
            }
            return result

        doc_result = next(iter(results.values()), None)
        if not doc_result:
            result = {
                "file": original_name,
                "success": False,
                "error": "results 中无文档内容"
            }
            return result

        # 查找 UUID 目录
        uuid_dirs = [d for d in os.listdir(ALL_OUT_DIR) if os.path.isdir(os.path.join(ALL_OUT_DIR, d))]
        uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
        uuid_dirs = [d for d in uuid_dirs if uuid_pattern.match(d)]

        if not uuid_dirs:
            result = {
                "file": original_name,
                "success": False,
                "error": f"未找到 UUID 输出目录: {ALL_OUT_DIR}"
            }
            return result

        # 取最新一个
        uuid_dir = max(uuid_dirs, key=lambda d: os.path.getctime(os.path.join(ALL_OUT_DIR, d)))
        uuid_output_dir = os.path.join(ALL_OUT_DIR, uuid_dir)
        uuid_dir_to_clean = uuid_output_dir  # 记录用于 finally 删除

        # 构造输出路径
        output_dir = os.path.join(ALL_OUT_DIR, uuid_dir, original_name, "auto")
        if not os.path.exists(output_dir):
            result = {
                "file": original_name,
                "success": False,
                "error": f"输出路径不存在: {output_dir}"
            }
            return result

        # 保存 Markdown
        md_output_path = os.path.join(output_dir, f"{original_name}.md")
        markdown_content = doc_result.get("md_content")
        if markdown_content:
            try:
                with open(md_output_path, "w", encoding="utf-8") as f:
                    f.write(markdown_content)
                print(f"✅ Markdown 已保存到: {md_output_path}")
            except Exception as e:
                print(f"❌ 保存 Markdown 失败: {e}")
        else:
            print(f"⚠️ 未生成 Markdown 内容")

        # 扁平化复制
        new_output_dir = os.path.join(ALL_OUT_DIR, original_name, "auto")
        os.makedirs(new_output_dir, exist_ok=True)

        for f in os.listdir(output_dir):
            if f.startswith(f"{original_name}_") and f.endswith(".json") or f.endswith(".md"):
                src = os.path.join(output_dir, f)
                dst = os.path.join(new_output_dir, f)
                shutil.copy2(src, dst)

        images_src = os.path.join(output_dir, "images")
        images_dst = os.path.join(new_output_dir, "images")
        if os.path.exists(images_src):
            shutil.copytree(images_src, images_dst, dirs_exist_ok=True)
        print(f"✅ 成功复制到扁平目录: {new_output_dir}")

        # 复制 middle.json 到 JSON_OUT_DIR
        middle_json_src = os.path.join(new_output_dir, f"{original_name}_middle.json")
        if os.path.exists(middle_json_src):
            os.makedirs(JSON_OUT_DIR, exist_ok=True)
            shutil.copy2(middle_json_src, json_output_path)
            print(f"✅ middle.json 已保存到: {json_output_path}")
        else:
            result = {
                "file": original_name,
                "success": False,
                "error": f"找不到 middle.json: {middle_json_src}"
            }
            return result

        md_src = os.path.join(new_output_dir, f"{original_name}.md")
        if os.path.exists(md_src):
            os.makedirs(MD_OUT_DIR, exist_ok=True)
            shutil.copy2(md_src, data_md_output_path)
            print(f"✅ Markdown 已保存到: {data_md_output_path}")
        else:
            result = {
                "file": original_name,
                "success": False,
                "error": f"找不到 md: {md_src}"
            }
            return result

        # ✅ 成功完成
        result = {
            "file": original_name,
            "success": True,
            "path": json_output_path,
            "error": None
        }
        return result

    except Exception as e:
        import traceback
        result = {
            "file": original_name,
            "success": False,
            "error": f"解析异常: {e}\n{traceback.format_exc()}"
        }
        return result  # 不在这里 return，交给 finally 处理后统一返回

    finally:
        # ✅ 无论成功失败，都尝试删除 UUID 临时目录
        if uuid_dir_to_clean and os.path.exists(uuid_dir_to_clean):
            try:
                shutil.rmtree(uuid_dir_to_clean)
                print(f"🗑️ 已删除临时目录: {uuid_dir_to_clean}")
            except PermissionError as e:
                print(f"⚠️ 删除失败（权限被占用）: {uuid_dir_to_clean} | {e}")
            except Exception as e:
                print(f"⚠️ 删除临时目录出错: {e}")

        # ✅ 删除原始 PDF 文件（可选）
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                print(f"🗑️ 已删除原始文件: {pdf_path}")
        except Exception as e:
            print(f"⚠️ 删除原始文件失败: {e}")



def parse_pdfs(pdf_list):
    """
    批量解析 PDF
    Args:
        pdf_list: PDF 文件名列表（如 ["a.pdf", "b.pdf"]）

    Returns:
        List[dict]: 解析结果列表，每个元素包含 file, success, error 等
    """
    results = []
    for pdf in pdf_list:
        input_path = os.path.join(INPUT_DIR, pdf)
        if not os.path.exists(input_path):
            results.append({
                "file": pdf,
                "success": False,
                "error": "文件不存在",
                "time": 0.0
            })
            continue

        start_time = time.time()  # ⏱️ 记录单个文件开始时间
        try:
            result = process_and_extract(input_path)
            end_time = time.time()
            elapsed = end_time - start_time

            if not isinstance(result, dict):
                result = {
                    "file": pdf,
                    "success": False,
                    "error": f"解析返回非字典: {result}",
                }
            result["time"] = round(elapsed, 2)  # 添加耗时字段
            results.append(result)
        except Exception as e:
            import traceback
            end_time = time.time()
            elapsed = end_time - start_time
            results.append({
                "file": pdf,
                "success": False,
                "error": f"解析时发生异常: {e}\n{traceback.format_exc()}",
                "time": round(elapsed, 2)
            })
    return results

# 示例调用
if __name__ == "__main__":
    pdf_path = "“土耳其模式”的新变化及其影响_王林聪.pdf"  # 替换为实际的 PDF 文件路径
    pdf_list = [pdf_path]
    parse_pdfs(pdf_list)