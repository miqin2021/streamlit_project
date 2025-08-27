import os
import time
import streamlit as st
from scripts.mineru_local_server import parse_pdfs  

INPUT_DIR = "data/uploads"
JSON_OUT_DIR = "data/json_layout"

# 整合所有CSS样式：保留文件列表隐藏 + 修复上传按钮空白
def inject_combined_css():
    combined_css = """
    <style>
    /* 保留：隐藏文件列表相关内容 */
    [data-testid="stFileUploader"] .st-emotion-cache-fis6aj,
    [data-testid="stFileUploader"] .st-emotion-cache-wbtvu4 {
        display: none !important;
    }

    /* 新增：修复上传按钮上方空白过大问题 */
    [data-testid="stFileUploader"] {
        margin-top: -2rem !important;  /* 减少顶部间距，可根据需要调整 */
        padding-top: 0 !important;
    }

    /* 调整上传区域内部元素间距 */
    [data-testid="stFileUploaderDropzone"] {
        margin-top: 0 !important;
        padding-top: 1rem !important;
    }

    /* 调整标题与上传组件的间距 */
    .stMarkdown + [data-testid="stFileUploader"] {
        margin-top: -1.5rem !important;
    }
    </style>
    """
    st.markdown(combined_css, unsafe_allow_html=True)

def render_right():
    st.markdown("#### 📚 上传文章")
    
    # 注入整合后的CSS
    inject_combined_css()
    
    # 文件上传组件
    uploaded_files = st.file_uploader(
        " ", 
        type=["pdf", "doc", "docx"],  
        accept_multiple_files=True, 
        key="uploader"
    )

    # 会话状态初始化
    deleted_set = st.session_state.setdefault("deleted_files", set())
    uploaded_set = st.session_state.setdefault("uploaded_set", set())
    # 新增：用于追踪文件是否被手动取消选中的状态
    unselected_files = st.session_state.setdefault("unselected_files", set())

    # 处理文件上传逻辑
    if uploaded_files:
        for f in uploaded_files:
            if f.name in deleted_set:
                continue
            save_path = os.path.join(INPUT_DIR, f.name)
            if not os.path.exists(save_path):
                with open(save_path, "wb") as out_file:
                    out_file.write(f.read())
                uploaded_set.add(f.name)
                # 新上传的文件默认选中，从取消选中集合中移除
                if f.name in unselected_files:
                    unselected_files.remove(f.name)
        st.session_state["uploaded_set"] = uploaded_set
        st.session_state["unselected_files"] = unselected_files

    # 展示已上传文件列表（带序号）
    SUPPORTED_EXTENSIONS = (".pdf", ".doc", ".docx")
    all_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(SUPPORTED_EXTENSIONS) and f not in deleted_set
    ])

    selected_files = []
    for idx, file in enumerate(all_files, start=1):
        base_name = os.path.splitext(file)[0]
        json_path = os.path.join(JSON_OUT_DIR, f"{base_name}.json")

        col1, col2, col3 = st.columns([6, 1, 2])
        col1.markdown(f"**{idx}. 📘 {file}**")  # 带序号展示

        with col2:
            # 关键修改：默认选中，除非在取消选中集合中
            is_checked = file not in unselected_files
            # 当用户取消选中时，添加到取消选中集合
            if not st.checkbox(
                "Select file", 
                key=f"check_{file}", 
                value=is_checked,
                label_visibility="hidden"
            ):
                unselected_files.add(file)
            else:
                # 如果用户重新选中，从取消选中集合中移除
                unselected_files.discard(file)
                selected_files.append(file)

        with col3:
            if st.button("删除", key=f"del_{file}"):
                try:
                    os.remove(os.path.join(INPUT_DIR, file))
                    if os.path.exists(json_path):
                        os.remove(json_path)
                    uploaded_set.discard(file)
                    deleted_set.add(file)
                    unselected_files.discard(file)  # 从取消选中集合中移除
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 删除失败：{e}")

    # 保存取消选中状态到会话
    st.session_state["unselected_files"] = unselected_files

    # 批量删除按钮和全部删除按钮
    if all_files:
        col_batch, col_all, col_null = st.columns([1, 1, 2])
        with col_batch:
            if st.button("🗑️ 批量删除选中文件"):
                if not selected_files:
                    st.warning("请先勾选要删除的文件！")
                else:
                    try:
                        for file in selected_files:
                            os.remove(os.path.join(INPUT_DIR, file))
                            json_path = os.path.join(JSON_OUT_DIR, f"{os.path.splitext(file)[0]}.json")
                            if os.path.exists(json_path):
                                os.remove(json_path)
                            uploaded_set.discard(file)
                            deleted_set.add(file)
                            unselected_files.discard(file)
                        st.success(f"✅ 成功删除 {len(selected_files)} 个文件！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 批量删除失败：{e}")
        with col_all:
            if st.button("🗑️ 全部删除"):
                try:
                    for file in all_files:
                        file_path = os.path.join(INPUT_DIR, file)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        json_path = os.path.join(JSON_OUT_DIR, f"{os.path.splitext(file)[0]}.json")
                        if os.path.exists(json_path):
                            os.remove(json_path)
                        uploaded_set.discard(file)
                        deleted_set.add(file)
                        unselected_files.discard(file)
                    st.success(f"✅ 成功删除所有 {len(all_files)} 个文件！")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 全部删除失败：{e}")

    # 解析按钮和底部导航
    if st.button("🚀 开始解析"):
        # 解析逻辑保持不变
        if "uploader" in st.session_state:
            del st.session_state["uploader"]
        st.session_state["deleted_files"] = set()

        files_to_parse = sorted([
            f for f in os.listdir(INPUT_DIR)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ])
        if not files_to_parse:
            st.warning("⚠️ 没有需要解析的文件！")
            return

        status_area = st.empty()
        progress_bar = st.progress(0.0)

        status_area.markdown(f"正在解析 {len(files_to_parse)} 篇文件...")
        results = parse_pdfs(files_to_parse)
        progress_bar.progress(1.0)

        success_count = sum(1 for r in results if r["success"])
        for r in results:
            if r["success"]:
                st.success(f"📘 {r['file']} ✅ 成功 (耗时：{r['time']:.2f}s)")
            else:
                st.error(f"📘 {r['file']} ❌ 失败 (耗时：{r['time']:.2f}s)：{r.get('error', '未知错误')}")
        st.session_state["parsed"] = (success_count == len(results))

        # 解析完成后，显示统计信息
        total_count = len(results)
        success_times = [r["time"] for r in results if r["success"]]

        if success_times:
            avg_time = sum(success_times) / len(success_times)
            max_time = max(success_times)
            fastest = min(success_times)
            st.caption(f"📊 成功 {success_count}/{total_count} 篇 | "
                    f"⏱️ 平均 {avg_time:.2f}s | "
                    f"最快 {fastest:.2f}s | "
                    f"最慢 {max_time:.2f}s")
        else:
            st.caption(f"📊 成功 {success_count}/{total_count} 篇")

    col_prev, col_next = st.columns([1, 1])
    with col_prev:
        if st.button("⬅️ 上一步"):
            st.session_state.step = 1
            st.rerun()
    with col_next:
        if st.button("➡️ 下一步"):
            if st.session_state.get("parsed", False):
                st.session_state.step = 3
                st.rerun()
            else:
                st.warning("请先解析文件！")

if __name__ == "__main__":
    # 初始化会话状态
    st.session_state.setdefault("uploaded_set", set())
    st.session_state.setdefault("deleted_files", set())
    st.session_state.setdefault("parsed", False)
    st.session_state.setdefault("unselected_files", set())  # 新增：追踪取消选中的文件
    render_right()