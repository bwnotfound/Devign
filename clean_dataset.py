import json
import os
from src.utils.functions.parse import tokenizer
from tqdm import tqdm
import configs
import argparse

PATHS = configs.Paths()

def process_chunk(chunk, chunk_idx):
    """处理一个数据块并生成新的 JSON"""
    new_json = []
    for data in tqdm(chunk, desc=f"Processing chunk {chunk_idx}"):
        code = data["func"]
        cleaned_code = " ".join(tokenizer(code))  # 清理代码
        new_json.append({"input": cleaned_code, "target": data["target"]})
    
    # 将处理后的 chunk 保存为临时文件
    output_path = os.path.join(PATHS.raw, "tem", f"dataset_cleaned_chunk_{chunk_idx}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_json, f, indent=4)

def clean_dataset(chunk_file, chunk_idx):
    """读取特定 chunk 文件并处理"""
    with open(chunk_file, "r", encoding="utf-8") as f:
        chunk = json.load(f)

    # 处理该数据块
    process_chunk(chunk, chunk_idx)

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Clean dataset chunk")
    parser.add_argument("--chunk_file", required=True, help="The chunk file to process")
    parser.add_argument("--chunk_idx", type=int, required=True, help="The index of the chunk")
    args = parser.parse_args()

    clean_dataset(args.chunk_file, args.chunk_idx)
