import os
import sys
import json
import time  # 新增：用于处理文件写入延迟
import streamlit as st
import pandas as pd
import subprocess
from io import BytesIO
import plotly.express as px

# 路径配置（与 step7_cluster_summary_LLM.py 对齐）
OUTPUT_DIR = "data/outputs"
OUTPUT_DIR_6 = os.path.join(OUTPUT_DIR, "step6")
OUTPUT_DIR_7 = os.path.join(OUTPUT_DIR, "step7")

# 文件路径（严格匹配脚本入参）
INPUT_JSON_STEP7 = os.path.join(OUTPUT_DIR_6, "step6_summary.json")  # Step6 输出
CLUSTER_OUTPUT_STEP7 = os.path.join(OUTPUT_DIR_7, "step7_summary.json")  # Step7 输出
PROMPT_INPUT_FILE_STEP7 = os.path.join(OUTPUT_DIR_7, "prompt_input.txt")  # 提示词文件
CLUSTER_SCRIPT = "scripts/step7_cluster_summary_LLM.py"  # 核心脚本

# 初始化 Session State（新增与脚本强相关的状态）
def init_session_state():
    if "step7_ran" not in st.session_state:
        st.session_state.step7_ran = False
    if "step7_error" not in st.session_state:
        st.session_state.step7_error = None
    if "step7_output" not in st.session_state:
        st.session_state.step7_output = None
    if "cluster_min" not in st.session_state:
        st.session_state.cluster_min = 2  # 聚类数量最小值默认
    if "cluster_max" not in st.session_state:
        st.session_state.cluster_max = 5  # 聚类数量最大值默认
    if "custom_prompt_step7" not in st.session_state:
        # 预设更专业的提示词（适配聚类总结任务）
        st.session_state.custom_prompt_step7 = (
            "任务：对下面来自同一类别内的句子再次进行总结。\n"
            "输入：{summaries}\n"
            "要求：\n"
            "1. 生成1条概括性的总结句；\n"
            "2. 语言学术、中性，避免主观评价；\n"
            "3. 直接输出总结，无需额外说明。"
        )
    if "summary_data_loaded_step7" not in st.session_state:
        st.session_state.summary_data_loaded_step7 = False
    if "step7_summary_data" not in st.session_state:
        st.session_state.step7_summary_data = None
    # 新增：记录当前处理的根类别，用于进度反馈
    if "current_root_class" not in st.session_state:
        st.session_state.current_root_class = ""

# 运行聚类脚本（优化进度反馈与错误解析）
def run_cluster_script(cluster_min: int, cluster_max: int, custom_prompt: str):
    os.makedirs(OUTPUT_DIR_7, exist_ok=True)
    
    # 保存提示词（确保脚本能读取到）
    with open(PROMPT_INPUT_FILE_STEP7, "w", encoding="utf-8") as f:
        f.write(custom_prompt.strip())
    
    # 检查 Step6 输入文件
    if not os.path.exists(INPUT_JSON_STEP7):
        st.error(f"未找到 Step6 结果文件：{INPUT_JSON_STEP7}")
        st.markdown("请先完成 **Step6（趋势总结与筛选）**！")
        return
    
    with st.spinner(f"开始聚类并进行二轮总结..."):
        try:
            # 构建命令（与脚本入参严格对齐）
            command = [
                sys.executable,
                CLUSTER_SCRIPT,
                "--input", INPUT_JSON_STEP7,
                "--output", CLUSTER_OUTPUT_STEP7,
                "--min_size", str(cluster_min),
                "--max_size", str(cluster_max),
                "--prompt_file", PROMPT_INPUT_FILE_STEP7
            ]
            
            # 执行命令（实时反馈根类别处理进度）
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 实时输出日志到 Streamlit（恢复 rerun 确保进度实时更新）
            st.session_state.step7_output = ""
            for line in process.stdout:
                st.session_state.step7_output += line
                st.rerun()  # 关键修复：取消注释，确保页面实时刷新
                if "Processing root class:" in line:
                    st.session_state.current_root_class = line.split(":")[-1].strip()
                    # 用 st.info 替代 st.write，避免多次输出导致界面混乱
                    st.info(f"🔍 正在处理根类别：{st.session_state.current_root_class}")
            
            # 等待执行完成（新增：延迟1秒，确保文件完全写入磁盘）
            returncode = process.wait(timeout=1800)
            time.sleep(1)  # 解决 IO 延迟导致的文件读取失败
            st.session_state.step7_error = process.stderr.read()
            
            if returncode == 0 and os.path.exists(CLUSTER_OUTPUT_STEP7):
                st.success("✅ 聚类总结完成！")
                # 重置加载状态，重新加载数据
                st.session_state.summary_data_loaded_step7 = False
                st.session_state.step7_summary_data = load_step7_data()
                # 强制刷新页面，触发结果展示
                st.rerun()
            else:
                st.error(f"❌ 脚本执行失败（退出码：{returncode}）")
                if "LLM API timeout" in st.session_state.step7_error:
                    st.warning("⚠️ 大模型调用超时，建议调整 prompt 或增大超时时间")
                st.code(st.session_state.step7_error, language="shell")
                
        except Exception as e:
            st.error(f"❌ 执行异常：{str(e)}")

# 加载 Step7 结果（优化错误提示，增加容错性）
def load_step7_data():
    
    # 尝试读取文件（增加重试机制，应对文件未完全写入）
    for _ in range(3):  # 最多重试3次
        try:
            # 以只读模式打开，确保文件未被占用
            with open(CLUSTER_OUTPUT_STEP7, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 宽松校验格式（避免过度严格导致正常数据被拒绝）
            if not isinstance(data, dict):
                st.error("Step7 结果格式错误：根节点应为字典类型")
                return None
            # 允许部分根类别值为空列表（避免个别空值导致整体失败）
            for root_class, clusters in data.items():
                if not isinstance(clusters, list):
                    st.warning(f"根类别 {root_class} 的值不是列表，已跳过该类别")
                    data[root_class] = []
            
            # 过滤空数据，确保至少有一个有效根类别
            valid_data = {k: v for k, v in data.items() if len(v) > 0}
            if not valid_data:
                st.warning("Step7 结果为空：所有根类别下均无聚类数据")
                return None
            
            st.session_state.step7_summary_data = valid_data
            st.session_state.summary_data_loaded_step7 = True
            return valid_data
        
        except json.JSONDecodeError as e:
            st.warning(f"读取 Step7 文件失败（JSON 格式错误），重试中...（{e}）")
            time.sleep(0.5)  # 重试前延迟0.5秒
        except Exception as e:
            st.error(f"读取 Step7 结果失败：{str(e)}")
            return None
    
    # 多次重试失败后提示
    st.error("Step7 文件读取失败：多次尝试后仍无法解析，请检查文件完整性")
    return None

# 渲染页面（强化交互与可视化）
def render_right():
    st.header("7️⃣ 二轮总结")
    init_session_state()
    
    # 检查 Step6 依赖
    if not os.path.exists(INPUT_JSON_STEP7):
        st.error("⚠️ 未找到 Step6 的总结数据")
        st.markdown("请先完成 **Step6** 并生成总结数据！")
        if st.button("⬅️ 返回 Step6", use_container_width=False):
            st.session_state.step = 6
            st.rerun()
        return
    
    st.markdown(
        "对 Step6 生成的根类别总结句进行聚类，通过设置**单聚类句子数量**控制子类粒度："
    )
    st.markdown(
        "<small>规则：若根类别总结句总数 ≤ 最大值，则不聚类；若总数 > 最大值，则自动聚类</small>",
        unsafe_allow_html=True
    )
    
    # 聚类参数设置
    col1, col2, col3 = st.columns([1, 1, 5])
    with col1:
        cluster_min = st.number_input(
            "最小值",
            min_value=2,
            max_value=49,
            value=st.session_state.cluster_min,
            step=1,
            help="单类最少句子数[2,49]"
        )
        st.session_state.cluster_min = cluster_min
    with col2:
        min_for_max = cluster_min + 1 if cluster_min + 1 <= 50 else 50
        cluster_max = st.number_input(
            "最大值",
            min_value=min_for_max,
            max_value=50,
            value=max(st.session_state.cluster_max, min_for_max),
            step=1,
            help="单类最多句子数[3,50]"
        )
        st.session_state.cluster_max = cluster_max

    # 参数说明折叠面板
    with st.expander("ℹ️ 查看完整参数说明", expanded=False):
        st.markdown("""
        #### 聚类核心逻辑
        通过设置「单聚类句子数量」，动态控制每个根类别的聚类行为：
        - **不聚类场景**：若根类别下的总结句总数 ≤ 「最大值」，直接保留为1个聚类（避免过度拆分）；
        - **聚类场景**：若总结句总数 > 「最大值」，自动拆分聚类，确保每个子类的句子数在 [最小值, 最大值] 范围内（保证聚类粒度均匀）。
        
        #### 示例
        - 设置：[2, 5]  
        → 根类别有4条总结句 → 不聚类（总数 ≤ 5） → 直接调用大模型进行二轮总结；  
        → 根类别有8条总结句 → 自动聚类为2个子类（每个子类4-5条句子） → 再对每个子类调用大模型进行二轮总结。
        """)

    # 自定义提示词
    custom_prompt = st.text_area(
        "请输入大模型用于总结的提示词（可选）：",
        value=st.session_state.custom_prompt_step7,
        height=200,
    )
    st.session_state.custom_prompt_step7 = custom_prompt
    
    # 执行聚类（强化状态反馈）
    if not st.session_state.step7_ran:
        if st.button("🚀 开始二轮总结", type="primary", use_container_width=False):
            st.session_state.step7_ran = True
            run_cluster_script(cluster_min, cluster_max, custom_prompt)
    else:
        # 实时展示处理进度（用 st.empty() 避免重复输出）
        progress_placeholder = st.empty()
        if st.session_state.current_root_class:
            progress_placeholder.info(f"🔍 正在处理根类别：{st.session_state.current_root_class}")
        else:
            progress_placeholder.info("✅ 聚类已完成，正在加载结果...")
        
        # 展示结果（强制重新加载数据，确保最新）
        summary_data = load_step7_data()
        # 清空进度提示
        progress_placeholder.empty()
        
        if not summary_data:
            st.warning("无法加载聚类结果，请检查脚本输出或点击「重新聚类」")
            if st.button("🔄 重新聚类", use_container_width=False):
                # 重置所有相关状态，避免残留
                st.session_state.step7_ran = False
                st.session_state.summary_data_loaded_step7 = False
                st.session_state.step7_summary_data = None
                st.session_state.current_root_class = ""
                st.rerun()
            return
        
        # -------------------------- 以下为结果展示逻辑（确保执行）--------------------------
        st.success("✅ 结果加载完成！")
        
        # 1. 结果统计
        total_root_classes = len(summary_data)
        cluster_counts = [len(clusters) for clusters in summary_data.values()]
        total_summaries = sum(cluster_counts)
        
        st.markdown("### 📊 聚类总结统计")
        stats_cols = st.columns(3)
        with stats_cols[0]:
            st.metric("根类别总数", total_root_classes)
        with stats_cols[1]:
            st.metric("平均聚类数/根类别", f"{total_summaries/total_root_classes:.1f}")
        with stats_cols[2]:
            st.metric("二轮总结总数", total_summaries)
        
        # 2. 聚类分布可视化

        df_cluster_counts = pd.DataFrame({
            "根类别": list(summary_data.keys()),
            "二轮总结数量": cluster_counts
        })
        st.markdown("### 📈 各根类别的二轮总结数量分布（按数量排序）")

        # 关键：按“二轮总结数量”降序排序，让图表更具可读性
        df_cluster_counts_sorted = df_cluster_counts.sort_values(by="二轮总结数量", ascending=False)

        fig = px.bar(
            df_cluster_counts_sorted,  # 使用排序后的DataFrame
            x="根类别",
            y="二轮总结数量",
            text="二轮总结数量",
            orientation="v",  # 垂直柱状图（与目标一致）
            title=None,       # 隐藏图表自带标题（用st.markdown单独控制标题）
            # 优化轴标签名称（更简洁易懂）
            labels={"二轮总结数量": "总结数量", "根类别": "类别"},
            color_discrete_sequence=["#636EFA"]  # 保持原配色
        )

        # 柱子上方显示数值（优化文本格式，避免重叠）
        fig.update_traces(
            texttemplate="%{text}",  # 仅显示数量数值
            textposition="outside",  # 数值在柱子外侧
            textfont=dict(size=10)   # 调整数值字体大小，避免拥挤
        )

        # 核心布局调整：隐藏Y轴、优化X轴、透明背景
        fig.update_layout(
            # X轴配置：保留类别标签，轻微倾斜避免重叠
            xaxis=dict(
                showticklabels=True,
                tickangle=-15,          # 标签倾斜-15度（比目标-45度更易读）
                tickfont=dict(size=11), # 调整类别标签字体大小
                showgrid=False          # 隐藏X轴方向网格线
            ),
            # Y轴配置：完全隐藏（包括刻度、标签、网格线）
            yaxis=dict(
                showticklabels=False,   # 隐藏Y轴数值标签
                showgrid=False,         # 隐藏Y轴方向网格线
                zeroline=False,         # 隐藏Y=0基准线
                visible=False           # 完全隐藏Y轴（包括轴线）
            ),
            # 图表整体样式
            height=500,                # 保持原高度
            width=800,                 # 固定宽度（与目标一致）
            showlegend=False,          # 隐藏图例（单一系列无需图例）
            plot_bgcolor='rgba(0,0,0,0)',  # 透明背景（与目标一致）
            margin=dict(l=20, r=20, t=40, b=60),  # 调整边距，避免内容被截断
            xaxis_title=None,          # 隐藏X轴标题（已在st.markdown中说明）
            yaxis_title=None           # 隐藏Y轴标题（已隐藏Y轴，无需标题）
        )

        st.plotly_chart(fig, use_container_width=True)
        
        # 3. 结果详情
        st.markdown("### 📋 二轮总结详情")
        # 确保 summary_data 是有效字典，遍历所有根类别
        for root_class, clusters in summary_data.items():
            with st.expander(f"📂 根类别：{root_class}（{len(clusters)} 条总结）", expanded=False):
                if not clusters:
                    st.caption("⚠️ 该根类别下无有效聚类数据")
                    continue
                for cluster in clusters:
                    # 确保 cluster 包含必要字段，避免 KeyError
                    if all(k in cluster for k in ["numbered_id", "summary", "count", "original_summaries"]):
                        # 高亮根类别关键词
                        root_keyword = root_class.split(".")[-1] if "." in root_class else root_class
                        highlighted_summary = cluster["summary"].replace(
                            root_keyword, 
                            f"<mark>{root_keyword}</mark>"
                        )
                        st.markdown(f"##### <span style='color:green'>{cluster['numbered_id']}. {highlighted_summary}</span>", unsafe_allow_html=True)
                        
                        # 展开查看原始内容
                        with st.expander("查看关联总结句", expanded=False):
                            for i, orig_summary in enumerate(cluster["original_summaries"], 1):
                                st.markdown(f"{i}. {orig_summary}")
                    else:
                        st.warning(f"聚类 {cluster.get('numbered_id', '未知')} 字段不完整，已跳过")
        
        # 4. 结果下载
        st.markdown("### 📥 下载结果")
        download_cols = st.columns(2)
        
        # JSON 下载
        json_data = json.dumps(summary_data, ensure_ascii=False, indent=2)
        with download_cols[0]:
            st.download_button(
                "JSON 格式",
                data=json_data,
                file_name="step7_summary.json",
                mime="application/json",
                use_container_width=False
            )
        
        # Excel 下载
        excel_buffer = BytesIO()
        excel_rows = []
        for root_class, clusters in summary_data.items():
            for cluster in clusters:
                if all(k in cluster for k in ["numbered_id", "summary", "count", "original_summaries"]):
                    excel_rows.append({
                        "根类别": root_class,
                        "总结编号": cluster["numbered_id"],
                        "二轮总结内容": cluster["summary"],
                        "原始 summary 数量": cluster["count"],
                        "原始 summary（前3条）": "|".join(cluster["original_summaries"][:3])
                    })
        
        if excel_rows:
            df_excel = pd.DataFrame(excel_rows)
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df_excel.to_excel(writer, sheet_name="二轮总结", index=False)
            
            with download_cols[1]:
                st.download_button(
                    "Excel 格式",
                    data=excel_buffer.getvalue(),
                    file_name="step7_summary.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=False
                )
        else:
            st.warning("无有效数据可生成 Excel 文件")
        
        # 5. 操作按钮
        st.markdown("### ⚙️ 操作")
        action_cols = st.columns(2)
        with action_cols[0]:
            if st.button("🔄 重新聚类总结", use_container_width=False):
                st.session_state.step7_ran = False
                st.session_state.summary_data_loaded_step7 = False
                st.session_state.step7_summary_data = None
                st.session_state.current_root_class = ""
                st.rerun()
        with action_cols[1]:
            if st.button("⬅️ 返回上一步", use_container_width=False):
                st.session_state.step = 6
                st.rerun()

if __name__ == "__main__":
    st.set_page_config(page_title="Step7 - 二轮总结", layout="wide")
    if "step" not in st.session_state:
        st.session_state.step = 7
    render_right()