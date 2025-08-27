# ============== 配置参数 ==============
TOKEN = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI1NTcwMzc5NiIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc1NTI0NTQzMSwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiMTU3MzAwNTQzMDYiLCJvcGVuSWQiOm51bGwsInV1aWQiOiI0MDdmYTQ4NS1mYTViLTQ2NjMtODZhZC0xOGE5M2Q0Y2Y5MjkiLCJlbWFpbCI6IiIsImV4cCI6MTc1NjQ1NTAzMX0.6eQjGzPWFAqxFmN9YR-WYZsADAvJfIv-PQtcSa01wPEHNkKwhHudmyvN8Z_y56l1Udm7idW6xBRawd-BU5PmtA"


import os
import time
import shutil
import zipfile
import requests
import streamlit as st

INPUT_DIR = "data/uploads"
OUTPUT_DIR = "data/output_md"
JSON_OUT_DIR = "data/json_layout"
GET_UPLOAD_URL_API = "https://mineru.net/api/v4/file-urls/batch"
CHECK_RESULT_API = "https://mineru.net/api/v4/extract-results/batch/"

# GET_UPLOAD_URL_API = "http://localhost:8815/file-urls/batch"
# CHECK_RESULT_API = "http://localhost:8815/extract-results/batch/"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(JSON_OUT_DIR, exist_ok=True)

# ====== 处理单个 PDF 并提取 JSON ======
def process_and_extract(pdf_path: str) -> str:
    file_name = os.path.basename(pdf_path)
    original_name = os.path.splitext(file_name)[0]
    result_dir = os.path.join(OUTPUT_DIR, original_name)
    os.makedirs(result_dir, exist_ok=True)
    zip_output_path = os.path.join(result_dir, f"{original_name}.zip")

    if os.path.exists(zip_output_path):
        return extract_layout_json(original_name)

    upload_url, batch_id = get_upload_url(file_name)

    if not upload_url:
        return None

    if not upload_file(upload_url, pdf_path):
        return None

    download_url = wait_for_extraction(batch_id)
    if not download_url:
        return None

    zip_path = os.path.join(result_dir, f"{original_name}.zip")
    if not download_result(download_url, zip_path):
        return None

    return extract_layout_json(original_name)

# ====== 调用 API ======
def get_upload_url(file_name):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {TOKEN}'
    }
    safe_file_name = file_name[:50]
    file_hash = hash(file_name) & 0xfffffff
    data_id = f"batch_{safe_file_name}_{file_hash}"

    data = {
        "files": [{
            "name": file_name,
            "is_ocr": True,
            "enable_formula": True,
            "enable_table": True,
            "language": "auto",
            "data_id": data_id
        }]
    }

    try:
        res = requests.post(GET_UPLOAD_URL_API, headers=headers, json=data)

        if res.status_code == 200:
            result = res.json()
            if result["code"] == 0:
                return result["data"]["file_urls"][0], result["data"]["batch_id"]
            else:
                print(f"[ERROR] 接口返回非 0 code: {result['code']} - {result.get('message')}")
        else:
            print(f"[ERROR] 请求失败，HTTP {res.status_code}")
    except Exception as e:
        st.warning(f"上传 URL 异常: {e}")
        print(f"[EXCEPTION] {e}")

    return None, None


def upload_file(upload_url, file_path):
    try:
        with open(file_path, 'rb') as f:
            response = requests.put(upload_url, data=f)
            return response.status_code == 200
    except Exception as e:
        st.warning(f"上传失败: {e}")
    return False

def wait_for_extraction(batch_id):
    check_url = f"{CHECK_RESULT_API}{batch_id}"
    headers = {'Authorization': f'Bearer {TOKEN}'}
    for _ in range(30):
        try:
            res = requests.get(check_url, headers=headers)
            if res.status_code == 200:
                result = res.json()
                if result["code"] != 0:
                    return None
                task = result["data"]["extract_result"][0]
                if task["state"] == "done":
                    return task["full_zip_url"]
                elif task["state"] == "failed":
                    return None
            time.sleep(10)
        except:
            time.sleep(10)
    return None

def download_result(zip_url, save_path):
    try:
        res = requests.get(zip_url, stream=True)
        if res.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in res.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
    except Exception as e:
        st.error(f"下载失败: {e}")
    return False

def extract_layout_json(name):
    zip_path = os.path.join(OUTPUT_DIR, name, f"{name}.zip")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()

            layout_candidates = [f for f in file_list if f.endswith('layout.json')]
            if not layout_candidates:
                print(f"[ERROR] zip 中找不到 layout.json")
                return None

            layout_file = layout_candidates[0]
            temp_dir = os.path.join(OUTPUT_DIR, name, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            zip_ref.extract(layout_file, temp_dir)
            output_file = os.path.join(JSON_OUT_DIR, f"{name}.json")
            shutil.copy2(os.path.join(temp_dir, layout_file), output_file)
            shutil.rmtree(temp_dir)
            return output_file
    except Exception as e:
        st.error(f"layout.json 提取失败: {e}")
    return None

def parse_pdfs(pdf_list):
    results = []
    for pdf in pdf_list:
        input_path = os.path.join(INPUT_DIR, pdf)
        if not os.path.exists(input_path):
            results.append({"file": pdf, "success": False, "error": "文件不存在"})
            continue

        try:
            layout_path = process_and_extract(input_path)
            if layout_path:
                results.append({"file": pdf, "success": True})
            else:
                results.append({"file": pdf, "success": False, "error": "未获取 layout.json"})
        except Exception as e:
            results.append({"file": pdf, "success": False, "error": str(e)})
    return results
