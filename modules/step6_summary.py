import os, re
import sys
import json
import streamlit as st
import pandas as pd
import subprocess
from io import BytesIO

OUTPUT_DIR = "data/outputs"
OUTPUT_DIR_4 = os.path.join(OUTPUT_DIR, "step4")
OUTPUT_DIR_5 = os.path.join(OUTPUT_DIR, "step5")
OUTPUT_DIR_6 = os.path.join(OUTPUT_DIR, "step6")  
OUTPUT_DIR_7 = os.path.join(OUTPUT_DIR, "step7")

# 外部脚本
SUMMARY_SCRIPT = "scripts/step6_summary_with_LLM.py"

# --- 配置 ---
INPUT_JSON_STEP4 = os.path.join(OUTPUT_DIR_4, "output-cluster-step4.json")    
INPUT_JSON_STEP5  = os.path.join(OUTPUT_DIR_5, "output-cluster-step5.json")
SUMMARY_OUTPUT = os.path.join(OUTPUT_DIR_5, "output-cluster-step5_summary.json") 

# 定义后端保存的文件路径（与前端下载文件名一致，便于对应）
BACKEND_JSON = os.path.join(OUTPUT_DIR_6, "step6_summary.json")  # 后端保存的JSON
BACKEND_CSV = os.path.join(OUTPUT_DIR_6, "step6_summary.csv")    # 后端保存的CSV
BACKEND_EXCEL = os.path.join(OUTPUT_DIR_6, "step6_summary.xlsx") # 后端保存的Excel

PROMPT_INPUT_FILE = os.path.join(OUTPUT_DIR_6, "prompt_input.txt")

# 初始化 session state（关键：添加 edited_summaries 的初始化）
def init_session_state():
    if "step6_ran" not in st.session_state:
        st.session_state.step6_ran = False
    if "step6_error" not in st.session_state:
        st.session_state.step6_error = None
    if "step6_output" not in st.session_state:
        st.session_state.step6_output = None
    if "summary_data_loaded" not in st.session_state:
        st.session_state.summary_data_loaded = False
    # 新增：初始化 edited_summaries（存储修改后的summary）
    if "edited_summaries" not in st.session_state:
        st.session_state.edited_summaries = {}  # 格式：{item_idx: 修改后的summary文本}
    # 新增：初始化选择动作（避免未定义报错）
    if "_select_action" not in st.session_state:
        st.session_state["_select_action"] = None
    # 新增：记录后端是否已保存文件（避免重复保存）
    if "step6_file_saved" not in st.session_state:
        st.session_state.step6_file_saved = False

def run_summary_script(custom_prompt: str):
    """运行总结脚本"""
    with st.spinner("正在调用大模型进行总结（可能需要几分钟）..."):
        try:
            with open(PROMPT_INPUT_FILE, "w", encoding="utf-8") as f:
                f.write(custom_prompt.strip())

            result = subprocess.run(
                [
                    sys.executable,
                    SUMMARY_SCRIPT,
                    "--input", INPUT_JSON_STEP4,
                    "--output", INPUT_JSON_STEP5,
                    "--prompt_file", PROMPT_INPUT_FILE
                ],
                capture_output=True,
                text=True,
                timeout=600
            )

            st.session_state.step6_ran = True
            st.session_state.step6_output = result.stdout
            st.session_state.step6_error = result.stderr
            st.session_state.step6_file_saved = False  # 生成新总结后，重置“文件已保存”状态

            if result.returncode == 0:
                st.success("✅ 总结生成完成！")
                st.session_state.summary_data_loaded = False
                # 生成新总结后，清空旧的编辑记录
                st.session_state.edited_summaries = {}
            else:
                st.error(f"❌ 脚本执行失败，退出码：{result.returncode}")
                st.code(result.stderr, language="shell")

        except Exception as e:
            st.error(f"❌ 执行失败：{e}")

def flatten_summary_tree(summary_tree, prefix=""):
    """展平总结树，添加唯一索引方便关联编辑"""
    flat = []
    def _flatten(current_tree, current_prefix, current_index):
        for key, value in current_tree.items():
            full_path = f"{current_prefix}.{key}" if current_prefix else key
            if isinstance(value, dict):
                if "summary" in value:
                    # 添加唯一索引，确保每个item能被准确关联
                    flat.append({
                        "idx": current_index,
                        "full_path": full_path,
                        "leaf_key": key,
                        "summary": value["summary"],
                        "sentences": value.get("sentences", []),
                        "count": value.get("count", 0)
                    })
                    current_index += 1
                else:
                    # 递归处理子节点
                    current_index = _flatten(value, full_path, current_index)
        return current_index
    # 初始索引从0开始
    _flatten(summary_tree, prefix, 0)
    return flat

def load_summary_data():
    """加载总结数据"""
    if "summary_data" in st.session_state and st.session_state.summary_data_loaded:
        return st.session_state.summary_data

    if not os.path.exists(SUMMARY_OUTPUT):
        st.warning(f"⚠️ 总结文件不存在：{SUMMARY_OUTPUT}")
        return None

    try:
        with open(SUMMARY_OUTPUT, "r", encoding="utf-8") as f:
            data = json.load(f)
        st.session_state.summary_data = data
        st.session_state.summary_data_loaded = True
        return data
    except json.JSONDecodeError:
        st.error("❌ 总结文件格式错误，无法解析JSON")
        return None
    except Exception as e:
        st.error(f"❌ 读取总结文件失败：{str(e)}")
        return None

# 新增：后端保存文件的工具函数（统一处理文件写入，避免重复代码）
def save_to_backend(data, file_path, file_type="json"):
    """
    后端保存文件到 Step6 输出目录
    参数：
        data: 待保存的数据（JSON字符串/CSV字符串/Excel Bytes）
        file_path: 后端保存的完整路径
        file_type: 文件类型（json/csv/excel）
    """
    try:
        if file_type == "json":
            # 保存JSON（确保格式化，便于后续读取）
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(data)
        elif file_type == "csv":
            # 保存CSV（utf-8-sig编码，支持中文）
            with open(file_path, "w", encoding="utf-8-sig") as f:
                f.write(data)
        elif file_type == "excel":
            # 保存Excel（Bytes流写入文件）
            with open(file_path, "wb") as f:
                f.write(data)
        return True  # 保存成功
    except Exception as e:
        st.warning(f"⚠️ 后端保存{file_type.upper()}文件失败：{str(e)}")
        return False  # 保存失败

def render_right():
    st.header("6️⃣ 一轮总结")
    # 第一步：初始化所有session_state变量（包括文件保存状态）
    init_session_state()

    # --- 自定义提示词 ---
    st.markdown("### 📝 调用大模型")
    default_prompt = (
        "以下是一些来自不同语境的词条，请基于它们的整体内容，"
        "从客观、语言分析的角度出发，提炼一个简洁的一句话总结，"
        "可包含事件背景（起因）、主要过程、以及最后的变化或影响（结果）。"
        "无需评论立场，请仅描述现象与事实。"
    )

    custom_prompt = st.text_area(
        label="请输入大模型用于总结的提示词（可选）：",
        value=default_prompt,
        height=150,
        label_visibility="visible"
    )

    # --- 按钮：生成总结 ---
    if not st.session_state.step6_ran:
        if st.button("🚀 开始总结", type="primary"):
            if not os.path.exists(INPUT_JSON_STEP4):
                st.error(f"❌ 聚类文件不存在（请先完成step4）")
            else:
                run_summary_script(custom_prompt)
                st.rerun()
    else:
        # 检查总结文件是否存在
        if not os.path.exists(SUMMARY_OUTPUT):
            st.error(f"❌ 总结文件未生成：{SUMMARY_OUTPUT}")
            st.session_state.step6_ran = False
            return

        # 加载总结数据
        summary_data = load_summary_data()
        if not summary_data:
            st.warning("⚠️ 无法加载总结数据，无法展示内容")
            return

        # 展平总结树（带唯一索引）
        flat_list = flatten_summary_tree(summary_data)
        if not flat_list:
            st.info("ℹ️ 无类别可展示（总结数据为空）")
            return

        st.markdown("### 📋 总结结果（可下载）")
        st.markdown(f"共找到 **{len(flat_list)}** 个子类")

        # --- 全选 / 清空 按钮 ---
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 全选", use_container_width=False):
                st.session_state["_select_action"] = "all"
                st.rerun()

        with col2:
            if st.button("🗑️ 清空", use_container_width=False):
                st.session_state["_select_action"] = "clear"
                st.rerun()

        # 处理选择动作（全选/清空）
        select_action = st.session_state["_select_action"]
        if select_action == "all":
            for idx in range(1, len(flat_list) + 1):
                st.session_state[f"keep_{idx}"] = True
            st.session_state["_select_action"] = None  # 重置动作
            st.rerun()

        elif select_action == "clear":
            for idx in range(1, len(flat_list) + 1):
                st.session_state[f"keep_{idx}"] = False
            st.session_state["_select_action"] = None  # 重置动作
            st.rerun()

        # 初始化默认选择状态（首次加载时全选）
        if not any(key.startswith("keep_") for key in st.session_state):
            for idx in range(1, len(flat_list) + 1):
                st.session_state[f"keep_{idx}"] = True

        # --- 渲染每一项（带编辑功能） ---
        kept_classes = []
        for display_idx, item in enumerate(flat_list, start=1):
            item_unique_idx = item["idx"]  # 从展平数据中获取唯一索引
            cols = st.columns([5, 1])

            with cols[0]:
                # 显示类别名称
                st.markdown(
                    f"##### <span style='color:green'>{item['leaf_key']}</span>",
                    unsafe_allow_html=True
                )
                # 可编辑的summary输入框：优先显示已修改的内容，无修改则显示原始内容
                default_summary = st.session_state.edited_summaries.get(
                    item_unique_idx,  # 用唯一索引匹配修改记录
                    item["summary"]   # 无修改时用原始summary
                )
                edited_summary = st.text_area(
                    label=f"📌 当前总结：",
                    value=default_summary,
                    height=80,
                    key=f"edit_summary_{item_unique_idx}",  # 唯一key避免冲突
                    # label_visibility="collapsed"  # 隐藏重复的label
                )

                # 保存修改到session_state（实时更新）
                st.session_state.edited_summaries[item_unique_idx] = edited_summary

                # 处理来源文件名清理（修复原代码中变量未定义问题）
                source_names = [s.get("source", "未知") for s in item["sentences"]]
                clean_names = [re.sub(r"_middle\.json$", "", name) for name in source_names]

                # 展开查看词条详情
                with st.expander(f"📄 查看该类别的词条（共 {len(item['sentences'])} 条）"):
                    for sent_idx, sentence in enumerate(item["sentences"], start=1):
                        # 匹配当前句子的清理后来源名
                        raw_source = sentence.get("source", "未知")
                        clean_name = re.sub(r"_middle\.json$", "", raw_source)
                        st.markdown(
                            f"**{sent_idx}.** {sentence['text']}  \n"
                            f"> 来源：`{clean_name}` | "
                            f"页码：`{sentence.get('page', '未知')}` | "
                            f"根类别：`{sentence.get('root_class', '未知')}`",
                            unsafe_allow_html=True
                        )

            with cols[1]:
                # 保留/取消保留的复选框
                if st.checkbox(
                    "保留",
                    key=f"keep_{display_idx}",
                    help=f"勾选以保留下载"
                ):
                    kept_classes.append({
                        "class": item["leaf_key"],
                        "summary": edited_summary,  
                        "sentences": [
                            {k: v for k, v in s.items() if k in ["text", "source", "page", "root_class"]}
                            for s in item["sentences"]
                        ]
                    })

        # --- 保存与下载（核心修改：新增后端保存逻辑） ---
        if not kept_classes:
            st.warning("⚠️ 请至少保留一个类别后再下载")
        else:
            st.success(f"✅ 已保留 {len(kept_classes)} 个类别")

            # 1. 生成JSON数据 + 后端保存
            json_data = json.dumps(kept_classes, ensure_ascii=False, indent=2)
            json_saved = save_to_backend(json_data, BACKEND_JSON, file_type="json")  # 后端保存JSON

            # 2. 生成CSV数据 + 后端保存
            expanded_csv_rows = []
            for cls in kept_classes:
                for sentence in cls["sentences"]:
                    expanded_csv_rows.append({
                        "类别名称": cls["class"],
                        "类别总结": cls["summary"],
                        "词条内容": sentence["text"],
                        "来源文件": sentence.get("source", "未知"),
                        "页码": sentence.get("page", "未知"),
                        "根类别": sentence.get("root_class", "未知")
                    })
            csv_df = pd.DataFrame(expanded_csv_rows)
            csv_data = csv_df.to_csv(index=False, encoding="utf-8-sig")  # utf-8-sig支持中文
            csv_saved = save_to_backend(csv_data, BACKEND_CSV, file_type="csv")  # 后端保存CSV

            # 3. 生成Excel数据 + 后端保存
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                # 工作表1：词条详情（包含所有字段）
                csv_df.to_excel(writer, sheet_name="词条详情", index=False)
                # 工作表2：类别概览（汇总信息）
                overview_df = pd.DataFrame([
                    {
                        "类别名称": item["class"],
                        "类别总结": item["summary"],
                        "词条数量": len(item["sentences"]),
                        "涉及来源文件": ", ".join({s.get("source", "未知") for s in item["sentences"]}),
                        "涉及根类别": ", ".join({s.get("root_class", "未知") for s in item["sentences"]})
                    }
                    for item in kept_classes
                ])
                overview_df.to_excel(writer, sheet_name="类别概览", index=False)
            excel_buffer.seek(0)  # 重置流指针，确保下载完整
            excel_data = excel_buffer.getvalue()
            excel_saved = save_to_backend(excel_data, BACKEND_EXCEL, file_type="excel")  # 后端保存Excel

            # 标记后端保存状态（全部保存成功才标记）
            if json_saved or csv_saved or excel_saved:
                st.session_state.step6_file_saved = True
                # st.success(f"✅ 后端文件已保存至：`{OUTPUT_DIR_6}`")  # 提示用户后端保存路径
            else:
                st.session_state.step6_file_saved = False

            # --- 下载按钮组（前端下载功能不变） ---
            st.markdown("### 📥 下载保留结果")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    label="JSON 格式",
                    data=json_data,
                    file_name="step6_summary.json",
                    mime="application/json",
                    use_container_width=False
                )
            with col2:
                st.download_button(
                    label="CSV 格式",
                    data=csv_data,
                    file_name="step6_summary.csv",
                    mime="text/csv",
                    use_container_width=False
                )
            with col3:
                st.download_button(
                    label="Excel 格式",
                    data=excel_data,
                    file_name="step6_summary.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=False
                )


    # --- 底部导航 ---
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ 上一步"):
            st.session_state.step = 5
            st.rerun()

    with col2:    
        # --- 重新生成总结 ---
        if st.button("🔄 重新生成"):
            # 重置所有与step6相关的状态（包括编辑记录和文件保存状态）
            st.session_state.step6_ran = False
            st.session_state.summary_data_loaded = False
            st.session_state.edited_summaries = {}  # 清空编辑记录
            st.session_state.step6_file_saved = False  # 重置文件保存状态
            for key in ["step6_output", "step6_error", "_select_action"]:
                st.session_state.pop(key, None)
            # 重置选择状态
            for key in list(st.session_state.keys()):
                if key.startswith("keep_"):
                    del st.session_state[key]
            st.rerun()

    if st.session_state.get("step6_ran"):
        with col3:
            if st.button("➡️ 下一步"):
                st.session_state.step = 7
                st.rerun()


# --- 主入口（单独运行时使用） ---
if __name__ == "__main__":
    st.set_page_config(page_title="Step 6 - 一轮总结", layout="wide")
    if "step" not in st.session_state:
        st.session_state.step = 6
    render_right()