# modules/step5_recursive.py
import os
import json
import streamlit as st
import subprocess
import sys
import pandas as pd
from io import BytesIO
OUTPUT_DIR = "data/outputs"
OUTPUT_DIR_4 = os.path.join(OUTPUT_DIR, "step4")
OUTPUT_DIR_5 = os.path.join(OUTPUT_DIR, "step5")
OUTPUT_DIR_6 = os.path.join(OUTPUT_DIR, "step6")
OUTPUT_DIR_7 = os.path.join(OUTPUT_DIR, "step7")

RECURSIVE_SCRIPT = "scripts/recursive_cluster_v8.py"  

INPUT_JSON = os.path.join(OUTPUT_DIR_4, "output-cluster-step4.json")
OUTPUT_JSON = os.path.join(OUTPUT_DIR_5, "output-cluster-step5.json")
OUTPUT_TREE = os.path.join(OUTPUT_DIR_5, "output-cluster-step5_tree.json")
OUTPUT_FILTERED = os.path.join(OUTPUT_DIR_5, "output-cluster-step5_filtered_out.json")
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR_5, "output-cluster-step5.xlsx")  


def flatten_tree_to_rows(tree, parent_path=""):
    """递归展平聚类树为表格行"""
    rows = []
    for class_name, content in tree.items():
        current_path = f"{parent_path}.{class_name}" if parent_path else class_name
        if isinstance(content, dict):
            rows.extend(flatten_tree_to_rows(content, current_path))
        elif isinstance(content, list):
            for item in content:
                rows.append({
                    "Class Path": current_path,
                    "Text": item.get("text", ""),
                    "Source": item.get("source", "未知"),
                    "Page": item.get("page", "未知"),
                    "RootClass": item.get("root_class", "未知")
                })
        else:
            # 其他情况（如 None）
            rows.append({
                "Class Path": current_path,
                "Text": "",
                "Source": "",
                "Page": "",
                "RootClass": ""
            })
    return rows


def save_to_excel(tree_data, excel_path):
    """将聚类结果保存为 Excel 文件"""
    rows = flatten_tree_to_rows(tree_data)
    df = pd.DataFrame(rows)
    df.to_excel(excel_path, index=False, engine='openpyxl')
    return excel_path


def render_right():
    st.header("5️⃣ 二轮聚类")

    # 用户可设置迭代聚类限制
    col1, col2 = st.columns([1, 3])  
    with col1:
        max_cluster_size = st.number_input(
            "二轮聚类单类最大词条数",
            min_value=2,
            max_value=50,
            value=20,
            step=1,
            help="令每个子类包含词条数量不超过此值时[2,50]，停止二轮聚类"
        )

    if True:  # 用于逻辑分组
        # st.markdown("### 🚀 执行聚类")
        col1, col2= st.columns([1, 1])  # 前两列放按钮，第三列空出

        with col1:
            if st.button("🔁 递归聚类"):
                with st.spinner("正在运行递归聚类 ..."):
                    try:
                        result = subprocess.run(
                            [
                                sys.executable,
                                RECURSIVE_SCRIPT,
                                "--input", INPUT_JSON,
                                "--output", OUTPUT_JSON,
                                "--max_cluster_size", str(max_cluster_size),
                                "--clustering-mode", "recursive"  # 显式指定
                            ],
                            capture_output=True,
                            text=True,
                        )

                        st.session_state["step5_ran"] = True
                        st.session_state["last_clustering_mode"] = "recursive"
                        if result.returncode == 0:
                            st.success("✅ 递归聚类完成！")
                            st.session_state["step5_output"] = result.stdout
                        else:
                            st.error(f"❌ 递归聚类失败，退出码：{result.returncode}")
                            st.code(result.stderr)
                    except Exception as e:
                        st.error(f"❌ 执行错误：{e}")

        with col2:
            if st.button("🌳 层次聚类"):
                with st.spinner("正在运行层次聚类 ..."):
                    try:
                        result = subprocess.run(
                            [
                                sys.executable,
                                RECURSIVE_SCRIPT,
                                "--input", INPUT_JSON,
                                "--output", OUTPUT_JSON,
                                "--max_cluster_size", str(max_cluster_size),
                                "--clustering-mode", "hierarchical"  # 指定层次聚类模式
                            ],
                            capture_output=True,
                            text=True,
                        )

                        st.session_state["step5_ran"] = True
                        st.session_state["last_clustering_mode"] = "hierarchical"
                        if result.returncode == 0:
                            st.success("✅ 层次聚类完成！")
                            st.session_state["step5_output"] = result.stdout
                        else:
                            st.error(f"❌ 层次聚类失败，退出码：{result.returncode}")
                            st.code(result.stderr)
                    except Exception as e:
                        st.error(f"❌ 执行错误：{e}")

    # --- 展示结果 ---
    if st.session_state.get("step5_ran") and os.path.exists(OUTPUT_TREE):
        with open(OUTPUT_TREE, "r", encoding="utf-8") as f:
            tree_data = json.load(f)
    
        st.markdown("### 聚类结构")
    
        for major_class, sub_tree in tree_data.items():
            with st.expander(f"{major_class}", expanded=False):
                for sub_class, sentences in sub_tree.items():
                    st.markdown(f"- {sub_class}")
    
        # --- 下载区域 ---
        st.markdown("---")
        st.markdown("### 📥 下载结果")

        col1, col2, col3 = st.columns(3)

        # JSON 下载
        with col1:
            if os.path.exists(OUTPUT_JSON):
                with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                    st.download_button(
                        "📥 下载 JSON",
                        f,
                        file_name=OUTPUT_JSON,
                        mime="application/json"
                    )

        # Excel 下载
        with col2:
            if os.path.exists(OUTPUT_JSON):
                try:
                    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                        full_data = json.load(f)
                    
                    # 生成 Excel 内容（BytesIO）
                    rows = flatten_tree_to_rows(full_data)
                    df = pd.DataFrame(rows)
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name="Cluster Results")
                    
                    # 下载按钮
                    st.download_button(
                        "📥 下载 Excel",
                        data=excel_buffer.getvalue(),
                        file_name=OUTPUT_EXCEL,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.warning(f"❌ 生成 Excel 失败: {e}")

        # 过滤日志下载
        with col3:
            if os.path.exists(OUTPUT_FILTERED):
                with open(OUTPUT_FILTERED, "r", encoding="utf-8") as f:
                    st.download_button(
                        "📥 过滤日志",
                        f,
                        file_name=OUTPUT_FILTERED,
                        mime="application/json"
                    )

    # --- 底部导航 ---
    col1, _, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ 上一步"):
            st.session_state.step = 4
            st.rerun()
    
    if st.session_state.get("step5_ran") and os.path.exists(OUTPUT_JSON):
        with col3:
            if st.button("➡️ 下一步"):
                st.session_state.step = 6
                st.rerun()


# --- 主入口 ---
if __name__ == "__main__":
    st.set_page_config(page_title="Step 5 - 二轮聚类", layout="wide")
    if "step" not in st.session_state:
        st.session_state.step = 5
    render_right()