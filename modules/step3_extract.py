# modules/step3_extract.py
import os
import urllib.parse
import streamlit as st
OUTPUT_DIR = "data/outputs"
step_dirs = [os.path.join(OUTPUT_DIR, f"step{i}") for i in range(4, 8)]
def create_output_directories():
    """批量创建所有步骤的输出目录，若已存在则跳过"""
    for dir_path in step_dirs:
        # 递归创建目录（如果父目录不存在也会自动创建）
        os.makedirs(dir_path, exist_ok=True)
        # 可选：打印创建信息（调试用）
        # print(f"已创建/确认目录: {dir_path}")

create_output_directories()
JSON_DIR = "data/json_layout"
SEE_PY_URL = "/see"

def render_right():
    st.header("3️⃣ 提取信息")

    if not os.path.exists(JSON_DIR):
        st.warning("📁 未找到解析后的 JSON 目录，请先完成步骤 2。")
        return

    json_files = sorted([f for f in os.listdir(JSON_DIR) if f.endswith(".json")])
    if not json_files:
        st.info("请先在步骤 2 中上传并解析 PDF 文件。")
        return

    if "step3_selected_files" not in st.session_state:
        st.session_state["step3_selected_files"] = {f: True for f in json_files}

    st.divider()
    for json_file in json_files:
        # 提取核心名称（去除"_middle.json"后缀）
        core_name = json_file.replace("_middle.json", "")        
        pdf_name = f"{core_name}.pdf"
        pdf_path = os.path.join("data/uploads", pdf_name)
        json_path = os.path.join(JSON_DIR, json_file)

        col1, col2, col3 = st.columns([6, 1, 2])
        col1.markdown(f"<div style='line-height: 2'>{core_name}</div>", unsafe_allow_html=True)
        default_checked = st.session_state["step3_selected_files"].get(json_file, True)
        st.session_state["step3_selected_files"][json_file] = col2.checkbox(
            " ", value=default_checked, key=f"chk_{json_file}"
        )

        if os.path.exists(pdf_path) and os.path.exists(json_path):
            see_url = f"{SEE_PY_URL}?pdf={urllib.parse.quote(pdf_path)}&json={urllib.parse.quote(json_path)}"
            preview_html = f"""
            <a href="{see_url}" target="_blank">
                <button style="margin-top: 2px;">预览</button>
            </a>
            """
            col3.markdown(preview_html, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ 上一步"):
            st.session_state.step = 2
            st.rerun()

    with col3:
        selected_files = [f for f, v in st.session_state["step3_selected_files"].items() if v]
        if selected_files and st.button("➡️ 下一步"):
            st.session_state.step = 4
            st.rerun()
