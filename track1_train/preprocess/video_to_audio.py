import os
import sys
import logging
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 日志系统配置
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dataset_conversion.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("AudioConverter")


class DatasetAudioExtractor:
    def __init__(self, dataset_root: str, output_root: str, max_workers: int = None):
        self.dataset_root = Path(dataset_root).resolve()
        self.output_root = Path(output_root).resolve()
        # 如果未指定线程数，默认使用 CPU 核心数 * 2
        self.max_workers = max_workers or (os.cpu_count() * 2)
        self.video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.webm'}
        self.target_sample_rate = "16000"

        # 验证环境依赖
        self._check_dependencies()

    def _check_dependencies(self):
        """验证系统是否安装了依赖工具"""
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("系统未检测到 ffmpeg。请确保 ffmpeg 已安装并加入到系统环境变量中。")
            sys.exit(1)

    def parse_qtype(self, filename_stem: str) -> str:
        if '_' not in filename_stem:
            return "unknown_qtype"

        parts = filename_stem.split('_')
        # 取最后一部分作为真正的类型（防止 ID 中自身带有下划线）
        qtype = parts[-1].strip()

        return qtype if qtype else "unknown_qtype"

    def process_single_video(self, video_path: Path, sub_set: str) -> bool:
        try:
            filename_stem = video_path.stem
            qtype = self.parse_qtype(filename_stem)
            # 构建分层目录：audio/qtype/train_val_test/
            target_dir = self.output_root / qtype / sub_set
            target_dir.mkdir(parents=True, exist_ok=True)

            output_file_path = target_dir / f"{filename_stem}.wav"

            # 组装 ffmpeg 参数
            # -y 强制覆盖，-vn 禁用视频，-ar 16k 采样，-ac 1 单声道，-c:a pcm_s16le 16bit无损封装
            command = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-vn',
                '-ar', self.target_sample_rate,
                '-ac', '1',
                '-c:a', 'pcm_s16le',
                str(output_file_path)
            ]

            # 执行系统命令
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg 转换失败: {video_path.name}。错误细节: {e.stderr.strip()}")
            return False
        except Exception as e:
            logger.error(f"处理文件时发生未知异常: {video_path.name}，原因: {str(e)}")
            return False

    def run(self):
        if not self.dataset_root.exists():
            logger.error(f"源数据集目录不存在: {self.dataset_root}")
            return

        sub_sets = ['train', 'val', 'test']
        tasks = []

        logger.info(f"开始扫描数据集文件...")
        # 1. 预先扫描所有待处理任务，便于计数和线程分配
        for sub_set in sub_sets:
            input_sub_path = self.dataset_root / sub_set
            if not input_sub_path.exists():
                logger.warning(f"跳过缺失子目录: {input_sub_path.name}")
                continue

            for file_path in input_sub_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.video_extensions:
                    tasks.append((file_path, sub_set))

        total_tasks = len(tasks)
        if total_tasks == 0:
            logger.warning("未在指定目录下找到匹配的视频文件。")
            return

        logger.info(f"共检测到 {total_tasks} 个视频文件。启动多线程线程池，并发数: {self.max_workers}。")

        # 2. 线程池并行处理
        success_count = 0
        failure_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务到线程池
            future_to_video = {
                executor.submit(self.process_single_video, video_path, sub_set): video_path
                for video_path, sub_set in tasks
            }
            # 动态获取处理结果
            for count, future in enumerate(as_completed(future_to_video), 1):
                video_path = future_to_video[future]
                try:
                    is_success = future.result()
                    if is_success:
                        success_count += 1
                    else:
                        failure_count += 1
                except Exception as exc:
                    logger.error(f"线程执行异常: {video_path.name}, 原因: {exc}")
                    failure_count += 1

                # 打印任务处理进度
                if count % 10 == 0 or count == total_tasks:
                    logger.info(f"进度进度: [{count}/{total_tasks}] | 成功: {success_count} | 失败: {failure_count}")

        logger.info("==========================================")
        logger.info(f"处理任务结束。总计: {total_tasks}, 成功: {success_count}, 失败: {failure_count}")
        logger.info(f"音频输出根目录为: {self.output_root}")
        logger.info("==========================================")


if __name__ == "__main__":
    # --------------------------------------------------
    # 路径配置（支持相对路径或绝对路径）
    # --------------------------------------------------
    DATASET_PATH = "./AVI_Challenge_dataset"
    OUTPUT_PATH = "./audio"

    # 实例化转换器并运行
    extractor = DatasetAudioExtractor(
        dataset_root=DATASET_PATH,
        output_root=OUTPUT_PATH,
        max_workers=8  # 可根据机器配置手动指定线程数，留空则自动计算
    )
    extractor.run()