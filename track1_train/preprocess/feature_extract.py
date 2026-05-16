import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from google import genai

# ==============================================================================
# 日志系统配置
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[sys.stdout, logging.FileHandler('gemini_pure_single_modality.log', encoding='utf-8')]
)
logger = logging.getLogger("PureSingleModality")

# ==============================================================================
# 基础配置与路径常量设置
# ==============================================================================
API_KEY = ""
MODEL_NAME = "gemini-embedding-2"

MAX_PER_MINUTE = 90
SLEEP_INTERVAL = 60 / MAX_PER_MINUTE
MAX_PER_DAY = 1000
TARGET_DIM = 1536

OUTPUT_BASE_DIR = "/data/home/chenqian/AVI2026/gemini_embeding"

DATASET_CONFIGS = [
    {
        "split_name": "train",
        "csv_path": "../train_data.csv",
        "video_dir": "../data/video/train_data",
        "audio_dir": "../data/audio_16k/train_data"
    },
    {
        "split_name": "val",
        "csv_path": "../val_data.csv",
        "video_dir": "../data/video/val_data",
        "audio_dir": "../data/audio_16k/val_data"
    },
    {
        "split_name": "test",
        "csv_path": "../test_data.csv",
        "video_dir": "../data/video/test_data",
        "audio_dir": "../data/audio_16k/test_data"
    }
]

# ==============================================================================
# 新增：问题类型（qtype）到具体标准问题的映射字典
# ==============================================================================
QTYPE_QUESTION_MAP = {
    "q1": "What would you consider among your greatest strengths and weaknesses as an employee?",
    "q2": "How would your best friend describe you?",
    "q3": "Think of situations when you made professional decisions that could affect your status or how much money "
          "you make. How do you usually behave in such situations? Why do you think that is?",
    "q4": "Think of situations when you joined a new team of people. How do you usually behave when you enter a new "
          "team? Why do you think that is?",
    "q5": "Think of situations when someone annoyed you. How do you usually react in such situations? Why do you "
          "think that is?",
    "q6": "Think of situations when your work or workspace were not very organized. How typical is that of you? Why "
          "do you think that is?"
}

GENDER_MAP = {1: "male", 2: "female"}
EDUCATION_MAP = {
    1: "less than high school education",
    2: "high school graduate",
    3: "some college education",
    4: "a Bachelor degree",
    5: "a Master's degree",
    6: "a Doctorate degree",
    7: "unspecified education level"
}

client = genai.Client(api_key=API_KEY)
processed_today = 0


# ==============================================================================
# 修改点：融合元信息与标准问题的文本构建函数
# ==============================================================================
def build_enriched_text(row, qtype):
    """
    将 CSV 中的元信息、根据 qtype 映射的标准问题、以及转录文本（content）重构为一个高密度的文本段落
    """
    # 1. 提取元信息并尝试映射转换（若无映射则保留原值）
    raw_gender = row.get("gender")
    gender_str = GENDER_MAP.get(raw_gender, str(raw_gender)) if pd.notna(raw_gender) else "unspecified gender"

    raw_edu = row.get("education")
    edu_str = EDUCATION_MAP.get(raw_edu, str(raw_edu)) if pd.notna(raw_edu) else "unspecified education level"

    age = row.get("age", "unknown")
    work_exp = row.get("work_experience", "unknown")
    question = QTYPE_QUESTION_MAP.get(qtype, "unknown interview question").strip()
    answer = str(row.get("content", "")).strip()
    # 4. 组装高密度 Context 文本
    enriched_prompt = (
        f"User Profile: [Gender: {gender_str}, Age: {age}, "
        f"Education: {edu_str}, Work Experience: {work_exp} years]. "
        f"Context: In a personality interview, the user was asked '{question}'. "
        f"User's Response: '{answer}'."
    )
    return enriched_prompt


def extract_and_save_feature(contents, qtype, modality_letter, split_name, filename, sample_id):
    global processed_today
    folder_name = f"gemini_embeddings_{qtype}_{modality_letter}"
    target_dir = os.path.join(OUTPUT_BASE_DIR, folder_name, split_name)
    os.makedirs(target_dir, exist_ok=True)
    save_path = os.path.join(target_dir, filename)
    if os.path.exists(save_path):
        return True

    if processed_today >= MAX_PER_DAY:
        logger.warning("已达到今日最高调用上限配额，提取中断。")
        return False
    try:
        result = client.models.embed_content(
            model=MODEL_NAME,
            contents=contents
        )
        embedding = np.array(result.embeddings[0].values, dtype=np.float32)
        np.savez(save_path, embedding=embedding, id=sample_id)
        processed_today += 1
        time.sleep(SLEEP_INTERVAL)
        return True
    except Exception as e:
        logger.error(f"提取样本 {sample_id} 的 [{modality_letter}] 模态时遭遇 API 错误: {e}")
        time.sleep(5)
        return False


# ==============================================================================
# 主执行流水线
# ==============================================================================
def main():
    global processed_today

    for config in DATASET_CONFIGS:
        split_name = config["split_name"]
        csv_path = config["csv_path"]
        video_dir = config["video_dir"]
        audio_dir = config["audio_dir"]

        if not os.path.exists(csv_path):
            logger.warning(f"缺失配置文件，跳过阶段: {csv_path}")
            continue

        logger.info(f"===> 开始处理数据集划分: {split_name} <===")
        df = pd.read_csv(csv_path)

        for idx, row in df.iterrows():
            if processed_today >= MAX_PER_DAY:
                logger.warning("达到每日上限，停止全局循环。")
                break

            sample_id = str(row.iloc[0])
            q_type = str(row.get("question_type", "unknown"))

            # ------------------------------------------------------------------
            # 修改点：动态调用 build_enriched_text 替代以前的 row.get("content")
            # ------------------------------------------------------------------
            content_text = build_enriched_text(row, q_type)

            npz_name = f"{sample_id}_{q_type}.npz"
            video_path = os.path.join(video_dir, f"{sample_id}_{q_type}.mp4")
            audio_path = os.path.join(audio_dir, f"{sample_id}_{q_type}.wav")

            if not os.path.exists(video_path) or not os.path.exists(audio_path):
                logger.warning(f"样本 {sample_id} 文件不完整，已跳过。")
                continue

            # 1. 文本单模态提取 (传入融合了元信息和问题的富文本)
            t_contents = [{"text": content_text}]
            logger.info(f"[{split_name}] 处理富文本模态 -> 样本: {sample_id} ({q_type})")
            if not extract_and_save_feature(t_contents, q_type, "t", split_name, npz_name, sample_id):
                if processed_today >= MAX_PER_DAY: break

            # 2. 音频单模态提取
            try:
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                a_contents = [{"inline_data": {"data": audio_bytes, "mime_type": "audio/wav"}}]
                logger.info(f"[{split_name}] 处理纯音频模态 -> 样本: {sample_id} ({q_type})")
                if not extract_and_save_feature(a_contents, q_type, "a", split_name, npz_name, sample_id):
                    if processed_today >= MAX_PER_DAY: break
            except Exception as e:
                logger.error(f"读取音频二进制失败 {sample_id}: {e}")

            # 3. 视频单模态提取
            try:
                with open(video_path, "rb") as f:
                    video_bytes = f.read()
                v_contents = [{"inline_data": {"data": video_bytes, "mime_type": "video/mp4"}}]
                logger.info(f"[{split_name}] 处理纯视频模态 -> 样本: {sample_id} ({q_type})")
                if not extract_and_save_feature(v_contents, q_type, "v", split_name, npz_name, sample_id):
                    if processed_today >= MAX_PER_DAY: break
            except Exception as e:
                logger.error(f"读取视频二进制失败 {sample_id}: {e}")

        if processed_today >= MAX_PER_DAY:
            break

    logger.info(f"所有任务执行完毕。今日成功请求 API 计数: {processed_today}")


if __name__ == "__main__":
    main()
