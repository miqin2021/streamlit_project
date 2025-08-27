import streamlit as st
import os, json

CACHE_FILE = "cache.json"


def render_right():
    # 初始化 query
    if "query" not in st.session_state:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
                st.session_state["query"] = data.get("query", "")
        else:
            st.session_state["query"] = ""

    # 渲染输入框
    st.markdown("#### 请输入你关注的问题")
    query = st.text_area(
        label="query",
        height=100,
        value=st.session_state["query"],
        label_visibility="collapsed"
    )
    # 保存
    if st.button("➡️ 下一步"):
        if query.strip() == "":
            st.warning("请先输入一个问题再继续。")
        else:
            st.session_state["query"] = query
            with open(CACHE_FILE, "w") as f:
                json.dump({"query": query}, f)

            st.session_state["step"] = 2
            st.rerun()


