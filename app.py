import streamlit as st
from modules.state_init import initialize_session_state, create_output_directories
from modules import (
    step1_query,
    step2_review,
    step3_extract,
    step4_clustering,
    step5_recurise_cluster,
    step6_summary,
    step7_summary,
)

from scripts.delete_files import delete_files
import os

# ======================== 核心优化：仅首次启动创建目录（用Session State标记）=======================
# 1. 先初始化Session State（必须在使用st.session_state前执行）
# 注意：initialize_session_state() 需放在最前面，确保状态变量可被后续逻辑使用
initialize_session_state()

# 2. 定义要创建的目录（原需求：step4~7）
OUTPUT_DIR = "data/outputs"
step_dirs = [os.path.join(OUTPUT_DIR, f"step{i}") for i in range(4, 8)]

# 3. 新增状态标记：判断是否已创建过目录（仅首次启动执行）
if "directories_created" not in st.session_state:
    # 首次启动：创建目录并打印日志
    try:
        create_output_directories(OUTPUT_DIR, step_dirs)
        print(f"✅ 初始化创建目录完成：{step_dirs}")  # 仅首次打印
        st.session_state["directories_created"] = True  # 标记为已创建
    except Exception as e:
        print(f"⚠️ 目录创建异常：{str(e)}")
        st.session_state["directories_created"] = False  # 标记为创建失败
else:
    # 非首次启动：跳过目录创建，不打印日志
    pass

st.set_page_config(layout="wide")

# 页面布局：左侧流程，右侧内容
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### 🧭 工作流程")

    # 核心步骤列表（不含清除按钮，保持流程纯净）
    steps = [
        ("1️⃣ 提出问题", step1_query),
        ("2️⃣ 检索文章", step2_review),
        ("3️⃣ 提取信息", step3_extract),
        ("4️⃣ 一轮聚类", step4_clustering),
        ("5️⃣ 二轮聚类", step5_recurise_cluster),
        ("6️⃣ 一轮总结", step6_summary),
        ("7️⃣ 二轮总结", step7_summary),  # 补充序号使格式统一
    ]

    current_step = st.session_state.get("step", 1)

    # 定义每一步是否完成的判断函数
    def is_step_completed(step_num):
        if step_num == 1:
            return bool(st.session_state.get("query", "").strip())
        elif step_num == 2:
            return bool(st.session_state.get("selected_pdfs"))
        elif step_num == 3:
            return "extracted_texts" in st.session_state
        elif step_num == 4:
            return "cluster_result" in st.session_state
        elif step_num == 5:
            return "summary_result" in st.session_state
        elif step_num == 6:
            return "final_summary" in st.session_state  # 补充二轮总结的完成判断
        return False

    # 渲染带状态的步骤按钮
    for idx, (label, _) in enumerate(steps, start=1):
        if idx < current_step:
            status = "✅"  # 已完成
        elif idx == current_step:
            status = "⏳"  # 进行中
        else:
            status = "⬜"  # 未开始

        # 步骤按钮点击逻辑
        if st.button(f"{status} {label}", key=f"nav_{idx}"):
            if idx <= current_step:
                st.session_state.step = idx
                st.rerun()
            elif idx == current_step + 1 and is_step_completed(current_step):
                st.session_state.step = idx
                st.rerun()
            else:
                st.warning(f"请先完成步骤 {current_step} 后再进入。")

    # ------------ 清除历史记录按钮（固定在左侧最下方）------------
    col_narrow, _ = st.columns([0.45, 0.55])  # 第一个列占 80% 宽度
    with col_narrow:
        st.markdown("---")
    # 使用危险样式按钮突出显示清除操作
    if st.button("🗑️ 清除历史记录", key="clear_history", type="secondary", use_container_width=False):
        delete_files()
        st.session_state.step = 1
        initialize_session_state()
        st.success("历史记录已清除，流程已重置")
        st.rerun()

# 右侧：主内容视图（每一步对应显示）
with col2:
    if st.session_state.step == 1:
        step1_query.render_right()
    elif st.session_state.step == 2:
        step2_review.render_right()
    elif st.session_state.step == 3:
        step3_extract.render_right()
    elif st.session_state.step == 4:
        step4_clustering.render_right()
    elif st.session_state.step == 5:
        step5_recurise_cluster.render_right()
    elif st.session_state.step == 6:
        step6_summary.render_right()
    elif st.session_state.step == 7:  
        step7_summary.render_right()