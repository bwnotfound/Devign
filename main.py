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
import shutil, sys, os
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
    is_w2v = False
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
        resume=True,
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


if __name__ == "__main__":
    # main()
    # create_task()
    # embed_task()
    embed_bert_task()
    # output_w2v_diff()
    # process_task(True, False)
