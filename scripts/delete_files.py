import os
import shutil
import streamlit as st
from modules.step1_query import CACHE_FILE

# 定义所有需要处理的目录（明确包含step4-step7，原step3未在列表中，若需清理需补充）
OUTPUT_DIR = "data/outputs"
step_dirs = [
    os.path.join(OUTPUT_DIR, "step4"),
    os.path.join(OUTPUT_DIR, "step5"),
    os.path.join(OUTPUT_DIR, "step6"),
    os.path.join(OUTPUT_DIR, "step7")
]

def is_directory_empty(path):
    """判断目录是否为空"""
    if not os.path.isdir(path):
        return False
    with os.scandir(path) as entries:
        return not any(entries)

def delete_files():
    # 1. 清理step1的缓存文件和session_state
    print("\n=== 清理step1的缓存文件 ===")
    if os.path.exists(CACHE_FILE):
        try:
            os.remove(CACHE_FILE)
            print(f"✅ 已删除 step1 缓存文件: {CACHE_FILE}")
        except Exception as e:
            print(f"❌ 删除 step1 缓存文件失败: {str(e)}")
    
    # 清空session_state中的query（仅当前函数内有效，建议在调用处重置更多状态）
    if "query" in st.session_state:
        del st.session_state["query"]
        print("✅ 已清空 session_state['query']")

    # 2. 核心修复：清理step4-step7目录内的所有文件和子目录（保留目录本身）
    print("\n=== 清理step4-step7目录内容 ===")
    for dir_path in step_dirs:
        print(f"处理目录: {dir_path}")
        
        # 仅处理存在的目录（跳过文件/不存在的路径）
        if not os.path.exists(dir_path):
            print(f"⚠️ 目录不存在，跳过: {dir_path}")
            continue
        if not os.path.isdir(dir_path):
            print(f"⚠️ 非目录项，跳过: {dir_path}")
            continue
        
        # 遍历目录内的所有文件/子目录，逐个删除
        for item_name in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item_name)
            
            try:
                # 处理文件/符号链接
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                    print(f"✅ 已删除文件: {item_path}")
                
                # 处理子目录（递归删除子目录内所有内容）
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"✅ 已删除子目录及内容: {item_path}")
            
            except Exception as e:
                print(f"❌ 删除 {item_path} 失败: {str(e)}")

    # 3. 清除Streamlit缓存
    print("\n=== 清除Streamlit缓存 ===")
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
        print("✅ 已清除所有Streamlit缓存")
    except Exception as e:
        print(f"❌ 清除缓存失败: {str(e)}")

    # 4. 可选：验证清理结果（打印目录是否为空）
    print("\n=== 清理结果验证 ===")
    for dir_path in step_dirs:
        if os.path.isdir(dir_path):
            if is_directory_empty(dir_path):
                print(f"✅ {dir_path} 已清空")
            else:
                print(f"⚠️ {dir_path} 仍有残留文件")
        else:
            print(f"⚠️ {dir_path} 不存在")