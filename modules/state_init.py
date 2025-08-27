import streamlit as st
import os
import shutil

# 定义路径
UPLOAD_DIR = "data/uploads"
ANNOTATION_DIR = "data/annotations"
JSON_DIR = "data/json_layout"
OUTPUT_DIR = "data/outputs"

step_dirs = [os.path.join(OUTPUT_DIR, f"step{i}") for i in range(4, 8)]

def create_output_directories(OUTPUT_DIR, subdirs):
    """
    创建目录：先创建根目录，再创建所有子目录
    :param root_dir: 根目录（如 data/outputs）
    :param subdirs: 子目录列表（如 [data/outputs/step4, ...]）
    """
    # 先创建根目录（若不存在）
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 再创建所有子目录（若不存在）
    for subdir in subdirs:
        os.makedirs(subdir, exist_ok=True)


if os.path.exists(JSON_DIR):
    shutil.rmtree(JSON_DIR)
os.makedirs(JSON_DIR, exist_ok=True)

def clear_directories():
    """清空上传和注释目录"""
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    if os.path.exists(ANNOTATION_DIR):
        shutil.rmtree(ANNOTATION_DIR)
    os.makedirs(ANNOTATION_DIR, exist_ok=True)

def initialize_session_state():
    """初始化 session_state 变量"""
    if 'selected_pdfs' not in st.session_state:
        st.session_state.selected_pdfs = set()
    if 'step' not in st.session_state:
        st.session_state.step = 1  # 当前步骤
    if 'retrieved_docs' not in st.session_state:
        st.session_state.retrieved_docs = []
    if 'selected_docs' not in st.session_state:
        st.session_state.selected_docs = []

# 每次启动 Streamlit 时都执行一次清理
if 'initialized' not in st.session_state:
    clear_directories()
    initialize_session_state()
    st.session_state.initialized = True
