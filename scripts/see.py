import streamlit as st
from pdf2image import convert_from_path
from PIL import ImageDraw, Image
import json
import os
import urllib.parse

import streamlit as st
import urllib.parse

pdf_path = st.query_params.get("pdf")
json_path = st.query_params.get("json")


# -------------------- 页面设置 --------------------
st.set_page_config(layout="wide", page_title="PDF 高亮查看器")

# -------------------- 从 URL 参数获取路径 --------------------
# query_params = st.experimental_get_query_params()
# pdf_path = query_params.get("pdf", [""])[0]
# json_path = query_params.get("json", [""])[0]

# # 路径解码（避免中文/空格问题）
# pdf_path = urllib.parse.unquote(pdf_path)
# json_path = urllib.parse.unquote(json_path)

# -------------------- 初始化高亮状态 --------------------
if "highlight_info" not in st.session_state:
    st.session_state.highlight_info = {"page": 1, "bbox": None}

# -------------------- 路径检查 --------------------
if not pdf_path or not os.path.exists(pdf_path):
    st.error(f"❌ PDF 文件不存在：{pdf_path}")
    st.stop()
if not json_path or not os.path.exists(json_path):
    st.error(f"❌ JSON 文件不存在：{json_path}")
    st.stop()

# -------------------- 加载 PDF 图片 --------------------
pdf_images = convert_from_path(pdf_path, dpi=150)

# -------------------- 加载 JSON 内容 --------------------
with open(json_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)

# -------------------- 页面布局 --------------------
col1, col2 = st.columns([2, 2])

# -------------------- 左侧 PDF 预览 --------------------
with col1:
    st.subheader("📄 PDF 预览")
    with st.container(height=800):
        for i, image in enumerate(pdf_images):
            page_number = i + 1
            img = image.copy().convert("RGBA")
            draw = ImageDraw.Draw(img, "RGBA")

            # 页面缩放比例
            scale_x = img.width / 595
            scale_y = img.height / 842

            # 若高亮当前页
            if st.session_state.highlight_info["page"] == page_number and st.session_state.highlight_info["bbox"]:
                bbox = st.session_state.highlight_info["bbox"]
                scaled_bbox = [
                    bbox[0] * scale_x,
                    bbox[1] * scale_y,
                    bbox[2] * scale_x,
                    bbox[3] * scale_y,
                ]
                draw.rectangle(scaled_bbox, fill=(255, 255, 0, 60), outline="red", width=3)

            st.image(img, caption=f"第 {page_number} 页", use_container_width=True)

# -------------------- 右侧 JSON 段落 --------------------
with col2:
    st.subheader("📝 JSON 段落内容")
    with st.container(height=800, border=True):
        for page in json_data.get("pdf_info", []):
            page_idx = page.get("page_idx", 0)
            page_num = page_idx + 1
            para_blocks = page.get("para_blocks", [])

            st.markdown(f"##### 📄 第 {page_num} 页")
            for block in para_blocks:
                if "lines" in block and block["lines"]:
                    paragraph_lines = []
                    for line in block["lines"]:
                        line_text = " ".join(span.get("content", "") for span in line.get("spans", []))
                        paragraph_lines.append(line_text.strip())
                    paragraph = " ".join(paragraph_lines).strip()

                    if not paragraph:
                        continue

                    preview = paragraph[:100] + "..." if len(paragraph) > 100 else paragraph
                    key = f"{page_num}-{block.get('index', id(block))}"
                    if st.button(preview, key=key):
                        st.session_state.highlight_info = {
                            "page": page_num,
                            "bbox": block.get("bbox")
                        }
            st.divider()
