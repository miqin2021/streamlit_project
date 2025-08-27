# modules/step4_clustering.py
import os
import json
import subprocess
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import time
OUTPUT_DIR = "data/outputs"
OUTPUT_DIR_4 = os.path.join(OUTPUT_DIR, "step4")
OUTPUT_DIR_5 = os.path.join(OUTPUT_DIR, "step5")
OUTPUT_DIR_6 = os.path.join(OUTPUT_DIR, "step6")
OUTPUT_DIR_7 = os.path.join(OUTPUT_DIR, "step7")

# --- 配置 ---
CLUSTER_SCRIPT_PATH = "scripts/cluster-v8.py"
OUTPUT_CLUSTER_JSON = os.path.join(OUTPUT_DIR_4, "output-cluster-step4.json")
OUTPUT_CLUSTER_EXCEL = os.path.join(OUTPUT_DIR_4, "output-cluster-step4.xlsx")
OUTPUT_IGNORED_TXT = os.path.join(OUTPUT_DIR_4, "output-cluster-step4-ignored-paragraphs.txt")
OUTPUT_IGNORED_EXCEL = os.path.join(OUTPUT_DIR_4, "output-cluster-step4-ignored-paragraphs.xlsx")

def render_right():
    st.header("4️⃣ 一轮聚类")

    # --- 原有代码：聚类数量设置、状态初始化 ---
    debug_mode = False
    col1, col2 = st.columns([1, 3])
    with col1:
        n_clusters = st.number_input(
            "⚙️ 设置聚类数量",
            min_value=2,
            max_value=50,
            value=9,
            step=1,
            help="指定要聚成多少个类别[2,50]"
        )

    if "step4_ran" not in st.session_state:
        st.session_state["step4_ran"] = False
    if "step4_success" not in st.session_state:
        st.session_state["step4_success"] = False

    # --- 关键修改：获取Step3选中的文件列表 ---
    # 从Step3的session_state中读取选中的JSON文件名
    selected_files = [f for f, v in st.session_state["step3_selected_files"].items() if v]
    if not selected_files:
        st.warning("⚠️ 请在Step3中至少选择一个文件")
        # 底部导航按钮禁用逻辑（可选）
        can_next = False
    else:
        can_next = True

    # --- 聚类按钮：传递选中文件列表 ---
    if not debug_mode:
        if st.button("🚀 开始聚类") and selected_files:  # 确保有选中文件才执行
            with st.spinner("正在聚类 ..."):
                start_time = time.time()

                # --- 关键修改：添加--selected_files参数，用逗号分隔文件名 ---
                command = [
                    sys.executable, 
                    CLUSTER_SCRIPT_PATH, 
                    "--n_clusters", str(n_clusters),
                    "--selected_files", ",".join(selected_files)  # 传递选中文件列表
                ]

                try:
                    result = subprocess.run(
                        command,  # 使用修改后的命令
                        capture_output=True,
                        text=True
                    )
                    # --- 原有代码：耗时计算、状态更新 ---
                    end_time = time.time()
                    duration = end_time - start_time
                    st.session_state["step4_ran"] = True
                    st.session_state["step4_output"] = result.stdout
                    st.session_state["step4_error"] = result.stderr
                    st.session_state["clustering_duration"] = duration
                    st.session_state["step4_success"] = (result.returncode == 0)

                    if result.returncode == 0:
                        st.success("✅ 聚类完成！")
                    else:
                        st.error(f"❌ 脚本执行失败，退出码：{result.returncode}")
                        st.code(result.stderr)
                except Exception as e:
                    # --- 原有异常处理代码 ---
                    end_time = time.time()
                    st.session_state["clustering_duration"] = end_time - start_time
                    st.session_state["step4_error"] = str(e)
                    st.error(f"❌ 执行失败：{e}")

    else:
        # 🔧 调试模式：直接模拟成功状态
        st.info("🔧 调试模式已启用：跳过聚类，直接加载已有结果")

        if st.button("✅ 模拟聚类完成"):
            # 假设耗时 2.5 秒
            st.session_state["step4_ran"] = True
            st.session_state["step4_success"] = True
            st.session_state["clustering_duration"] = 2.5  # 模拟耗时
            st.success("✅ 已进入调试模式，加载预存结果")
            # 注意：后续逻辑会检查 OUTPUT_CLUSTER_JSON 是否存在

    # ✅ 只有执行后才展示聚类结果
    if st.session_state.get("step4_ran"):
        cluster_json_exists = os.path.exists(OUTPUT_CLUSTER_JSON)
        cluster_excel_exists = os.path.exists(OUTPUT_CLUSTER_EXCEL)
        ignored_txt_exists = os.path.exists(OUTPUT_IGNORED_TXT)
        ignored_excel_exists = os.path.exists(OUTPUT_IGNORED_EXCEL)

        # --- 展示聚类分布统计表 ---
        if cluster_json_exists:
            try:
                with open(OUTPUT_CLUSTER_JSON, "r", encoding="utf-8") as f:
                    cluster_data = json.load(f)

                # 处理Category，移除开头的序号（如"1."、"2."等）
                cluster_counts = [
                    {
                        "Category": k.split(".", 1)[1] if "." in k else k,  # 分割一次并取后半部分
                        "Flat Entry Count": len(v)
                    } 
                    for k, v in cluster_data.items()
                ]

                df_cluster_stats = pd.DataFrame(cluster_counts).sort_values(
                    by="Flat Entry Count", ascending=False
                ).reset_index(drop=True)

                total_paragraphs = df_cluster_stats["Flat Entry Count"].sum()
                num_clusters = len(df_cluster_stats)

                # 获取耗时（保留两位小数）
                duration = st.session_state.get("clustering_duration", 0)
                duration_str = f"{duration:.1f}s"

                # === 指标卡片 ===
                st.markdown("### 📊 聚类概览")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 总段落数", total_paragraphs)
                with col2:
                    st.metric("📁 聚类总数", num_clusters)
                with col3:
                    st.metric("⏱️ 聚类耗时", duration_str)  

                # === 2. 柱状图可视化 ===
                st.markdown("### 📈 聚类分布统计（按数量排序）")
                fig = px.bar(
                    df_cluster_stats,
                    x="Category",
                    y="Flat Entry Count",
                    text="Flat Entry Count",
                    orientation="v",
                    title=None,  # 隐藏标题
                    labels={"Flat Entry Count": "词条数量", "Category": "类别"},
                    color_discrete_sequence=["#636EFA"]
                )

                # 柱子上方显示数值
                fig.update_traces(
                    texttemplate="%{text}",
                    textposition="outside"
                )

                # 关键：隐藏 Y 轴刻度、刻度标签、网格线
                fig.update_layout(
                    xaxis_title=None,           # 隐藏 X 轴标题
                    yaxis_title=None,           # 隐藏 Y 轴标题
                    yaxis=dict(
                        showticklabels=False,   # ❌ 隐藏左侧数字（Y轴刻度标签）
                        showgrid=False,         # ❌ 隐藏水平横线（网格线）
                        zeroline=False,         # ❌ 隐藏 Y=0 的轴线（可选）
                        visible=False           # 完全隐藏 Y 轴（包括刻度和标签）
                    ),
                    xaxis=dict(
                        showticklabels=True,    # 保留 X 轴类别标签
                        tickangle=-15,          # X轴标签倾斜，避免重叠
                    ),
                    height=500,
                    width=800,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',  # 透明背景（可选）
                    margin=dict(l=20, r=20, t=40, b=60)  # 调整边距
                )

                st.plotly_chart(fig, use_container_width=True)

                # # === 3. 可交互表格（支持搜索、排序）===
                # st.markdown("### 📋 详细统计表")
                # # 添加搜索框
                # search_term = st.text_input("🔍 搜索类别名称：", "")
                # if search_term:
                #     df_filtered = df_cluster_stats[df_cluster_stats["Category"].str.contains(search_term, case=False)]
                # else:
                #     df_filtered = df_cluster_stats

                # st.dataframe(df_filtered, use_container_width=True)

                # === 4. 多格式下载 ===
                try:
                    from openpyxl import Workbook
                    from io import BytesIO
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_cluster_stats.to_excel(writer, index=False, sheet_name="Cluster Stats")
                    excel_data = output.getvalue()
                    st.download_button(
                        label="📥 下载 Excel",
                        data=excel_data,
                        file_name="cluster_summary_stats.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except ImportError:
                    st.warning("⚠️ 请安装 openpyxl 以启用 Excel 下载：`pip install openpyxl`")

            except Exception as e:
                st.warning(f"⚠️ 读取聚类统计数据失败：{e}")

        # --- 展示聚类结果文件 ---
        if cluster_json_exists or cluster_excel_exists:
            st.subheader("📊 聚类结果")
            info_col, json_col, excel_col = st.columns([3, 1, 1])
            with info_col:
                if cluster_json_exists:
                    try:
                        with open(OUTPUT_CLUSTER_JSON, 'r', encoding='utf-8') as f:
                            cluster_data = json.load(f)
                        num_clusters = len(cluster_data)
                        total_paragraphs = sum(len(paras) for paras in cluster_data.values())
                        st.write(f"📦 共生成 {num_clusters} 个聚类，包含 {total_paragraphs} 个段落")
                    except Exception as e:
                        st.warning(f"⚠️ 读取聚类 JSON 出错: {e}")
                elif cluster_excel_exists:
                    try:
                        df = pd.read_excel(OUTPUT_CLUSTER_EXCEL)
                        total_rows = len(df)
                        st.write(f"📦 聚类结果 Excel 共 {total_rows} 行")
                    except Exception as e:
                        st.warning(f"⚠️ 读取 Excel 出错: {e}")

            if cluster_json_exists:
                with json_col:
                    with open(OUTPUT_CLUSTER_JSON, 'r', encoding='utf-8') as f:
                        st.download_button("📥 JSON", f, file_name=OUTPUT_CLUSTER_JSON, mime="application/json")

            if cluster_excel_exists:
                with excel_col:
                    with open(OUTPUT_CLUSTER_EXCEL, 'rb') as f:  # ✅ 关键：只用 rb
                        st.download_button(
                            "📥 Excel",
                            f,
                            file_name=OUTPUT_CLUSTER_EXCEL,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

        # --- 跳过段落日志展示 ---
        if ignored_txt_exists or ignored_excel_exists:
            st.subheader("📤 跳过内容")
            info_col, txt_col, excel_col = st.columns([3, 1, 1])
            with info_col:
                if ignored_txt_exists:
                    with open(OUTPUT_IGNORED_TXT, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    st.write(f"📄 跳过词条共 {len(lines)} 条")

            if ignored_txt_exists:
                with txt_col:
                    with open(OUTPUT_IGNORED_TXT, 'r', encoding='utf-8') as f:
                        st.download_button("📥 TXT", f, file_name=OUTPUT_IGNORED_TXT)

            if ignored_excel_exists:
                with excel_col:
                    with open(OUTPUT_IGNORED_EXCEL, 'rb') as f:
                        st.download_button("📥 Excel", f, file_name=OUTPUT_IGNORED_EXCEL)

        if not any([cluster_json_exists, cluster_excel_exists, ignored_txt_exists, ignored_excel_exists]):
            st.info("ℹ️ 聚类执行完成，但未找到输出文件。请检查脚本。")

    # --- 底部导航：仅当有选中文件且聚类成功时显示下一步 ---
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ 上一步"):
            st.session_state.step = 3
            st.rerun()
    with col3:
        # 关键修改：用can_next确保有选中文件
        if st.session_state.get("step4_ran") and os.path.exists(OUTPUT_CLUSTER_JSON) and can_next:
            if st.button("➡️ 下一步"):
                st.session_state.step = 5
                st.rerun()

# --- 主入口 ---
if __name__ == "__main__":
    st.set_page_config(page_title="Step 4 - 一轮聚类", layout="wide")
    if "step" not in st.session_state:
        st.session_state.step = 4
    render_right()