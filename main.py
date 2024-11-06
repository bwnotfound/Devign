# -*- coding: utf-8 -*-
"""
    This module is intended to join all the pipeline in separated tasks
    to be executed individually or in a flow by using command-line options

    Example:
    Dataset embedding and processing:
        $ python taskflows.py -e -pS
"""
import faulthandler

faulthandler.enable()

import argparse
import gc
import shutil
import shutil, sys, os, json
from argparse import ArgumentParser
from datetime import datetime

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from gensim.models.word2vec import Word2Vec

import configs
import src.data as data
import src.prepare as prepare
import src.process as process
import src.utils.functions.cpg as cpg
from src.utils.objects.input_dataset import InputDataset

from tqdm import tqdm
import torch


CREATE_PARAMS = configs.Create()
PROCESS_PARAMS = configs.Process()
PATHS = configs.Paths()
FILES = configs.Files()
DEVICE = PROCESS_PARAMS.device


def select(dataset):
    # dataset = dataset.loc[dataset["project"] == "FFmpeg"]
    len_filter = dataset.func.str.len() < 1200
    dataset = dataset.loc[len_filter]
    # print(len(dataset))
    # dataset = dataset.iloc[11001:]
    # print(len(dataset))
    # 暂时只使用前200条数据
    if CREATE_PARAMS.data_size != -1:
        dataset = dataset.head(CREATE_PARAMS.data_size)

    return dataset


def setup():
    os.makedirs(PATHS.cpg, exist_ok=True)
    os.makedirs(PATHS.model, exist_ok=True)
    os.makedirs(PATHS.input, exist_ok=True)
    os.makedirs(PATHS.tokens, exist_ok=True)
    os.makedirs(PATHS.w2v, exist_ok=True)


def create_task():
    # 将sys.stdout重定向到文件logs.txt
    use_cache = True
    clean = False
    sys.stdout = open('create_logs.txt', 'w', encoding='utf-8', buffering=1)
    if os.path.exists(PATHS.cpg) and not use_cache:
        shutil.rmtree(PATHS.cpg)
        os.mkdir(PATHS.cpg)
    if os.path.exists(PATHS.joern):
        shutil.rmtree(PATHS.joern)
        os.mkdir(PATHS.joern)
    if use_cache:
        cpg_names = os.listdir(PATHS.cpg)
        cpg_names = [cpg_name for cpg_name in cpg_names if cpg_name.endswith(".bin")]
        cpg_names.sort(key=lambda x: int(x.split("_")[0]))
        if len(cpg_names) > 0:
            os.remove(os.path.join(PATHS.cpg, cpg_names[-1]))
    context = configs.Create()
    raw = data.read(PATHS.raw, FILES.raw)
    print(len(raw))
    filtered = data.apply_filter(raw, select)
    filtered = data.clean(filtered)
    print(len(filtered))
    data.drop(filtered, ["commit_id", "project"])
    slices = data.slice_frame(filtered, context.slice_size)
    slices = [(s, slice.apply(lambda x: x)) for s, slice in slices]
    print(len(slices))

    cpg_files = []
    # Create CPG binary files
    for s, slice in slices:
        if use_cache:
            if os.path.exists(os.path.join(PATHS.cpg, f"{s}_{FILES.cpg}.bin")):
                cpg_files.append(f"{s}_{FILES.cpg}.bin")
                continue
        data.to_files(slice, PATHS.joern)
        cpg_file = prepare.joern_parse(
            context.joern_cli_dir, PATHS.joern, PATHS.cpg, f"{s}_{FILES.cpg}"
        )
        cpg_files.append(cpg_file)
        time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{time_stamp}: Dataset {s} to cpg.")
        shutil.rmtree(PATHS.joern)
    # Create CPG with graphs json files
    exists = [
        os.path.exists(os.path.join(PATHS.cpg, f"{s}_{FILES.cpg}.json"))
        for s, _ in slices
    ]
    if not (use_cache and all(exists)):
        json_files = prepare.joern_create(
            context.joern_cli_dir,
            PATHS.cpg,
            PATHS.cpg,
            cpg_files,
            use_cache=use_cache,
        )
    # json_files = [f"{s}_{FILES.cpg}.json" for s, _ in slices]
    for (s, slice), json_file in zip(slices, json_files):
        if not os.path.exists(os.path.join(PATHS.cpg, json_file)):
            print(
                f"Dataset chunk {s} not processed. Most likely due to Joern error json not parse successfully."
            )
            continue
        graphs = prepare.json_process(
            PATHS.cpg, json_file
        )  # 去除一些无用的字符串信息，并使用文件数字对每个graph进行编号
        if graphs is None:
            print(f"Dataset chunk {s} not processed.")
            continue
        dataset = data.create_with_index(
            graphs, ["Index", "cpg"]
        )  # 用上面的index对每一行数据创建索引
        dataset = data.inner_join_by_index(slice, dataset)  # 同时将分块的索引写入每一行
        print(f"Writing cpg dataset chunk {s}.")
        data.write(dataset, PATHS.cpg, f"{s}_{FILES.cpg}.pkl")
        if clean and os.path.exists(os.path.join(PATHS.cpg, f"{s}_{FILES.cpg}.pkl")):
            if os.path.exists(os.path.join(PATHS.cpg, f"{s}_{FILES.cpg}.bin")):
                os.remove(os.path.join(PATHS.cpg, f"{s}_{FILES.cpg}.bin"))
            if os.path.exists(os.path.join(PATHS.cpg, f"{s}_{FILES.cpg}.json")):
                os.remove(os.path.join(PATHS.cpg, f"{s}_{FILES.cpg}.json"))
        del dataset
        gc.collect()


def embed_task():

    context = configs.Embed()
    # Tokenize source code into tokens
    dataset_files = data.get_directory_files(
        PATHS.cpg
    )  # 从data/cpg文件家中取出之前生成好的所有.pkl文件
    w2vmodel = Word2Vec(**context.w2v_args)
    w2v_init = True

    tokens = []
    for pkl_file in tqdm(dataset_files, desc="Building w2v model"):
        file_name = pkl_file.split(".")[0]
        cpg_dataset = data.load(PATHS.cpg, pkl_file)
        tokens_dataset = data.tokenize(
            cpg_dataset
        )  # 对程序源码文本进行分词，返回分词的结果
        data.write(tokens_dataset, PATHS.tokens, f"{file_name}_{FILES.tokens}")
        for i in range(len(tokens_dataset)):
            tokens.append(tokens_dataset.tokens.iloc[i])
        # word2vec used to learn the initial embedding of each token
    tqdm.write("Building w2v model.")
    w2vmodel.build_vocab(tokens)
    tqdm.write("Training w2v model.")
    w2vmodel.train(tokens, total_examples=w2vmodel.corpus_count, epochs=context.epochs)

    for pkl_file in tqdm(dataset_files):
        file_name = pkl_file.split(".")[0]
        cpg_dataset = data.load(PATHS.cpg, pkl_file)
        # word2vec used to learn the initial embedding of each token

        if w2v_init:
            w2v_init = False
        # Embed cpg to node representation and pass to graph data structure
        cpg_dataset["nodes"] = cpg_dataset.apply(
            lambda row: cpg.parse_to_nodes(row.cpg, context.max_nodes),
            axis=1,  # context.max_nodes限定了对多的节点个数，不足的补0，超过的会截断
        )
        # remove rows with no nodes
        cpg_dataset = cpg_dataset.loc[cpg_dataset.nodes.map(len) > 0]
        cpg_dataset["input"] = cpg_dataset.apply(
            lambda row: prepare.nodes_to_input(
                row.nodes,
                row.target,
                context.max_nodes,
                w2vmodel.wv,
                context.edge_type,
                w2vmodel.wv.vector_size,
            ),
            axis=1,
        )  # 使用w2vec对每一个节点的文本进行编码
        data.drop(cpg_dataset, ["nodes"])
        tqdm.write(f"Saving input dataset {file_name} with size {len(cpg_dataset)}.")
        data.write(
            cpg_dataset[["input", "target"]], PATHS.input, f"{file_name}_{FILES.input}"
        )
        del cpg_dataset
        gc.collect()
    print("Saving w2vmodel.")
    w2vmodel.save(f"{PATHS.w2v}/{FILES.w2v}")


@torch.no_grad()
def embed_bert_task():
    from transformers import RobertaTokenizer, AutoTokenizer, RobertaModel, AutoModel
    import pickle
    from time import perf_counter

    context = configs.Embed()
    # Tokenize source code into tokens
    dataset_files = data.get_directory_files(
        PATHS.cpg
    )  # 从data/cpg文件家中取出之前生成好的所有.pkl文件

    # if not os.path.exists("data/tokens.pkl"):
    #     tokens = []
    #     for pkl_file in tqdm(dataset_files, desc="Building w2v model"):
    #         file_name = pkl_file.split(".")[0]
    #         cpg_dataset = data.load(PATHS.cpg, pkl_file)
    #         tokens_dataset = data.tokenize(
    #             cpg_dataset
    #         )  # 对程序源码文本进行分词，返回分词的结果
    #         data.write(tokens_dataset, PATHS.tokens, f"{file_name}_{FILES.tokens}")
    #         for i in range(len(tokens_dataset)):
    #             tokens.append(tokens_dataset.tokens.iloc[i])
    #         # word2vec used to learn the initial embedding of each token
    #     with open("data/tokens.pkl", "wb") as f:
    #         pickle.dump(tokens, f)
    # else:
    #     with open("data/tokens.pkl", "rb") as f:
    #         tokens = pickle.load(f)

    device = "cuda"
    tokenizer: RobertaTokenizer = AutoTokenizer.from_pretrained(
        'FacebookAI/roberta-base', add_prefix_space=True
    )
    model: RobertaModel = AutoModel.from_pretrained('FacebookAI/roberta-base').to(
        device
    )
    # flat_tokens = [token for sublist in tokens for token in sublist]
    # flat_tokens = [[t] for t in list(set(flat_tokens))[:1000]]

    # tokenized_input: list[list[str]] = tokenizer(
    #     flat_tokens, return_tensors='pt', is_split_into_words=True, padding=True
    # )
    # tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}

    # output = model(**tokenized_input)
    # now = perf_counter()
    # output = model(**tokenized_input)
    # print(f"Took {perf_counter() - now:.3f}s to run the model")
    # # 取出最后一层的hidden_states的第一个token的embedding
    # result = output.last_hidden_state[:, 0, :].cpu().detach().numpy()

    # print(result.shape)
    # print(result)

    # print(len(tokenized_input))
    # # print tokenizer cls_token and sep_token
    # print(tokenizer.cls_token, tokenizer.sep_token)
    # # print index==0 token
    # print(tokenizer.convert_ids_to_tokens(0))
    # print(tokenized_input)

    # exit()
    def tokenize(word_list: list[str]):
        word_list = [[w] for w in word_list]
        tokenized_input: list[list[str]] = tokenizer(
            word_list, return_tensors='pt', is_split_into_words=True, padding=True
        )
        tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}
        # tqdm.write(f"Start embed {len(word_list)} words.")
        # now = perf_counter()
        output = model(**tokenized_input)
        # tqdm.write(f"Took {perf_counter() - now:.3f}s to run the model")
        result = output.last_hidden_state[:, 0, :].cpu().detach().numpy()
        return result

    for pkl_file in tqdm(dataset_files):
        file_name = pkl_file.split(".")[0]
        cpg_dataset = data.load(PATHS.cpg, pkl_file)

        cpg_dataset["nodes"] = cpg_dataset.apply(
            lambda row: cpg.parse_to_nodes(row.cpg, context.max_nodes),
            axis=1,  # context.max_nodes限定了对多的节点个数，不足的补0，超过的会截断
        )
        # remove rows with no nodes
        cpg_dataset = cpg_dataset.loc[cpg_dataset.nodes.map(len) > 0]
        cpg_dataset["input"] = cpg_dataset.apply(
            lambda row: prepare.nodes_to_input(
                row.nodes,
                row.target,
                context.max_nodes,
                tokenize,
                context.edge_type,
                model.config.hidden_size,  # 需要设置768 + 1=769
                is_tokenize_func=True,
            ),
            axis=1,
        )  # 使用w2vec对每一个节点的文本进行编码
        data.drop(cpg_dataset, ["nodes"])
        tqdm.write(f"Saving input dataset {file_name} with size {len(cpg_dataset)}.")
        data.write(
            cpg_dataset[["input", "target"]], PATHS.input, f"{file_name}_{FILES.input}"
        )
        del cpg_dataset
        gc.collect()


@torch.no_grad()
def output_w2v_diff():
    is_w2v = True
    w2vmodel = Word2Vec.load(f"{PATHS.w2v}/{FILES.w2v}")
    vocab = list(w2vmodel.wv.index_to_key)[:100]
    if is_w2v:
        words_similarity = []

        # 查看特定词汇向量与其他词的相似度
        for word in tqdm(vocab):
            similarity_scores = []
            for other_word in vocab:
                similarity = w2vmodel.wv.similarity(word, other_word)
                similarity_scores.append(float(similarity))

            words_similarity.append(similarity_scores)
    else:
        from transformers import (
            RobertaTokenizer,
            AutoTokenizer,
            RobertaModel,
            AutoModel,
        )
        from time import perf_counter

        device = "cuda"
        tokenizer: RobertaTokenizer = AutoTokenizer.from_pretrained(
            'FacebookAI/roberta-base', add_prefix_space=True
        )
        vocab = list(tokenizer.get_vocab().keys())[:100]
        model: RobertaModel = AutoModel.from_pretrained('FacebookAI/roberta-base').to(
            device
        )
        tokens_input = [[t] for t in vocab]

        tokenized_input: list[list[str]] = tokenizer(
            tokens_input, return_tensors='pt', is_split_into_words=True, padding=True
        )
        tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}

        now = perf_counter()
        output = model(**tokenized_input)
        print(f"Took {perf_counter() - now:.3f}s to run the model")
        # 取出最后一层的hidden_states的第一个token的embedding
        # result = output.last_hidden_state[:, 0, :]
        result = torch.mean(output.last_hidden_state[:, 1:, :], dim=1)
        c = torch.nn.CosineSimilarity(dim=-1)
        # words_similarity = torch.matmul(result, result.T)
        words_similarity = c(result.unsqueeze(1), result.unsqueeze(0))
        words_similarity = words_similarity.cpu().detach().numpy()

    from matplotlib import pyplot as plt
    import numpy as np

    words_similarity = np.array(words_similarity)
    # 打印为黑白图, x轴为从小到大，从左到右，y轴为从小到大，从下到上
    plt.figure(figsize=(10, 10))
    plt.imshow(words_similarity, cmap='gray_r')
    plt.gca().xaxis.set_ticks_position("top")  # 将x轴刻度移到顶部
    plt.gca().xaxis.set_label_position("top")  # 将x轴标签移到顶部
    plt.show()


def process_task(stopping, test_only=False):
    context = configs.Process()
    devign = configs.Devign()
    model_path = PATHS.model + FILES.model
    model = process.Devign(
        path=model_path,
        device=DEVICE,
        model=devign.model,
        max_nodes=configs.Embed().max_nodes,
        learning_rate=devign.learning_rate,
        weight_decay=devign.weight_decay,
        loss_lambda=devign.loss_lambda,
        resume=False,
    )
    train = process.Train(model, context.epochs)
    input_dataset = data.loads(PATHS.input, PROCESS_PARAMS.dataset_ratio)
    # split the dataset and pass to DataLoader with batch size
    train_loader, val_loader, test_loader = list(
        map(
            lambda x: x.get_loader(context.batch_size, True),
            data.train_val_test_split(input_dataset, False),
        )
    )
    print(f"Train iter: {len(train_loader)}, dataset size: {len(train_loader.dataset)}")
    train_loader_step = process.LoaderStep("Train", train_loader, DEVICE)
    val_loader_step = process.LoaderStep("Validation", val_loader, DEVICE)
    test_loader_step = process.LoaderStep("Test", test_loader, DEVICE)

    if test_only:
        model.load()
        process.predict(model, test_loader_step)
        return

    print(f"train with {DEVICE}.")
    if stopping:
        early_stopping = process.EarlyStopping(
            model,
            patience=context.patience,
            verbose=PROCESS_PARAMS.verbose,
            delta=PROCESS_PARAMS.delta,
        )
        train(train_loader_step, val_loader_step, early_stopping)
        model.load()
    else:
        train(train_loader_step, val_loader_step)
        model.save()

    process.predict(model, test_loader_step)


def main():
    """
    main function that executes tasks based on command-line options
    """
    parser: ArgumentParser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--prepare', help='Prepare task', required=False)
    parser.add_argument("-c", "--create", action="store_true")
    parser.add_argument("-e", "--embed", action="store_true")
    parser.add_argument("-p", "--process", action="store_true")
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument("-pS", "--process_stopping", action="store_true")

    args = parser.parse_args()

    setup()
    if args.create:
        create_task()
    if args.embed:
        embed_task()
    if args.process:
        process_task(False)
    if args.process_stopping:
        process_task(True)
    if args.test:
        process_task(False, True)


# def clean_dataset():
#     import json
#     from src.utils.functions.parse import tokenizer

#     with open(os.path.join(PATHS.raw, "dataset_plus.json"), "r", encoding="utf-8") as f:
#         dataset_json = json.load(f)
#     new_json = []
#     for data in tqdm(dataset_json):
#         code = data["func"]
#         cleaned_code = "".join(tokenizer(code))
#         new_json.append({"input": cleaned_code, "target": data["target"]})
#     with open(
#         os.path.join(PATHS.raw, "dataset_cleaned.json"), "w", encoding="utf-8"
#     ) as f:
#         json.dump(new_json, f, indent=4)

def gpt_main():
    import json
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
    from datasets import load_dataset
    import torch
    from torch.utils.data import DataLoader

    # GPT use for classification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ratio = 0.8
    batch_size = 16
    eval_only = True
    resume = True
    prompt = "Please check the following code and determine if it has any potential bug: \n"

    try:
        if not resume:
            raise RuntimeError("Should not resume")
        resume_dir = "output"
        tokenizer = AutoTokenizer.from_pretrained(resume_dir)
        model = AutoModelForSequenceClassification.from_pretrained(resume_dir, num_labels=2)
    except Exception as e:
        print(str(e))
        print("No available checkpoint found, using pretrained model.")
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125m')
        model = AutoModelForSequenceClassification.from_pretrained('EleutherAI/gpt-neo-125m', num_labels=2)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # load dataset
    with open(
        os.path.join(PATHS.raw, "dataset_plus_cleaned.json"), "r", encoding="utf-8"
    ) as f:
        dataset = json.load(f)[:10000]
    dataset = [{"input": prompt + s["input"], "target": s["target"]} for s in dataset]
    train_size = int(len(dataset) * train_ratio)
    from random import shuffle

    shuffle(dataset)
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    with open(
        os.path.join(PATHS.raw, "train_dataset.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(train_dataset, f, indent=4)
    with open(
        os.path.join(PATHS.raw, "test_dataset.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(test_dataset, f, indent=4)
    train_dataset = load_dataset('json', data_files=os.path.join(PATHS.raw, "train_dataset.json"))
    test_dataset = load_dataset('json', data_files=os.path.join(PATHS.raw, "test_dataset.json"))

    train_dataset = train_dataset["train"]
    test_dataset = test_dataset["train"]

    def tokenize_function(examples):
        # return pt
        result = tokenizer(
            examples["input"],
            truncation=True,
            max_length=512,
            return_tensors="pt",
            padding="max_length",
        )
        labels = examples["target"]
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return {**result, "labels": labels_tensor}

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', "labels"])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', "labels"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    model.train()
    model.to(device)
    
    if not eval_only:
        for epoch in range(4):
            t_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}")
            total_loss = 0
            correct, total = 0, 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                output = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = output.loss
                loss.backward()
                optimizer.step()
                correct += (output.logits.argmax(dim=-1) == labels).sum().item()
                total += len(labels)
                acc = correct / total
                total_loss += loss.item()
                t_bar.set_postfix_str(f"Loss: {total_loss / total:.3f}, Acc: {acc:.3f}")
                t_bar.update()
            model.save_pretrained("output")

    model.eval()
    correct, total = 0, 0
    confuse_matrix = [[0, 0] for _ in range(2)]
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            output = model(input_ids, attention_mask=attention_mask, labels=labels)
            correct += (output.logits.argmax(dim=-1) == labels).sum().item()
            total += len(labels)
            for i in range(len(labels)):
                confuse_matrix[labels[i].item()][output.logits[i].argmax(dim=-1).item()] += 1
    acc = correct / total
    print(f"Test accuracy: {acc:.3f}")
    print("Confuse matrix:")
    print(confuse_matrix)

def clean_dataset_multiprocess():
    import subprocess
    # 获取原始数据集文件路径
    dataset_path = os.path.join(PATHS.raw, "dataset_small.json")
    
    num_chunks = os.cpu_count() - 1  # 获取可用的 CPU 核心数减去 2，作为进程数
    chunk_size = os.path.getsize(dataset_path) // num_chunks  # 每个 chunk 的大小（可以根据需要调整）

    # Step 1: 将数据集分割为多个文件
    print(f"Splitting dataset into {num_chunks} chunks..., every chunk size: {chunk_size}")
    os.makedirs(os.path.join(PATHS.raw, "chunks"), exist_ok=True)
    os.makedirs(os.path.join(PATHS.raw, "tem"), exist_ok=True)

    # 读取整个数据集并将其分割成多个较小的 chunk 文件
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 将数据集按 chunk 数量分割
    chunked_datasets = []
    for i in range(num_chunks):
        chunk = dataset[i::num_chunks]  # 选择每个 chunk
        chunk_file = os.path.join(PATHS.raw, "chunks", f"dataset_chunk_{i}.json")
        with open(chunk_file, "w", encoding="utf-8") as out_file:
            json.dump(chunk, out_file, indent=4)
        chunked_datasets.append(chunk_file)

    # Step 2: 使用 subprocess 启动多个进程执行 clean_dataset
    processes = []
    for i, chunk_file in enumerate(chunked_datasets):
        # 调用 clean_dataset.py 并传递 chunk 文件和其索引作为参数
        command = [
            sys.executable, "clean_dataset.py", "--chunk_file", chunk_file, "--chunk_idx", str(i)
        ]
        process = subprocess.Popen(command)
        processes.append(process)

    # 等待所有进程完成
    for process in processes:
        process.communicate()

    # Step 3: 合并所有处理后的结果
    print("Merging chunks into one cleaned dataset...")
    combined_json = []
    for i in range(num_chunks):
        chunk_file = os.path.join(PATHS.raw, "tem", f"dataset_cleaned_chunk_{i}.json")
        with open(chunk_file, "r", encoding="utf-8") as f:
            combined_json.extend(json.load(f))
        os.remove(chunk_file)
    shutil.rmtree(os.path.join(PATHS.raw, "tem"))
    shutil.rmtree(os.path.join(PATHS.raw, "chunks"))

    # 将合并后的数据保存为最终的 cleaned 数据集
    with open(os.path.join(PATHS.raw, "dataset_small_cleaned.json"), "w", encoding="utf-8") as f:
        json.dump(combined_json, f, indent=4)

if __name__ == "__main__":
    # main()
    # create_task()
    # embed_task()
    # embed_bert_task()
    output_w2v_diff()
    # process_task(True, False)
    # gpt_main()
