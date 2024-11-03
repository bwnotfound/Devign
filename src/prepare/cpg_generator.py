import json
import re
import subprocess
import os.path
import os
import time
from threading import Timer

from .cpg_client_wrapper import CPGClientWrapper

#from ..data import datamanager as data
def kill_command(p):
    """终止命令的函数"""
    p.kill()

def funcs_to_graphs(funcs_path):
    client = CPGClientWrapper()
    # query the cpg for the dataset
    print(f"Creating CPG.")
    graphs_string = client(funcs_path)
    # removes unnecessary namespace for object references
    graphs_string = re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", '', graphs_string)
    graphs_json = json.loads(graphs_string)

    return graphs_json["functions"]


def graph_indexing(graph):
    idx = int(graph["file"].split("/")[-1].split(".c")[0])
    graph.pop("file")
    return idx, {"functions": [graph]}


def joern_parse(joern_path, input_path, output_path, file_name):
    out_file = file_name + ".bin"
    joern_parse_call = subprocess.run([joern_path + "joern-parse", input_path, "--out", output_path + out_file],
                                      stdout=subprocess.PIPE, text=True, check=True)
    print(str(joern_parse_call))
    return out_file

def process_joern_file(joern_path, in_path, out_path, cpg_file):
    joern_process = subprocess.Popen([joern_path + "joern"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    json_file_name = f"{cpg_file.split('.')[0]}.json"

    # print(in_path+cpg_file)  
    if os.path.exists(in_path+cpg_file):
        # 设置定时器去终止这个命令
        # timer = Timer(30, kill_command, [joern_process])
# importCpg("/mnt/c/Users/86150/Desktop/图神经网络/devign_lab-main/data/cpg/0_cpg.bin")
# cpg.runScript("/mnt/c/Users/86150/Desktop/图神经网络/devign_lab-main/joern/graph-for-funcs.sc").toString() |> "/mnt/c/Users/86150/Desktop/图神经网络/devign_lab-main/test.json"
# cpg.runScript("/root/bin/joern//graph-for-funcs.sc").toString() |> "/mnt/c/Users/86150/Desktop/图神经网络/devign_lab-main/test.json"
        json_out = f"{os.path.abspath(out_path)}/{json_file_name}"
        import_cpg_cmd = f"importCpg(\"{os.path.abspath(in_path)}/{cpg_file}\")\r".encode()
        script_path = f"{os.path.dirname(os.path.abspath(joern_path))}/graph-for-funcs.sc"
        run_script_cmd = f"cpg.runScript(\"{script_path}\").toString() |> \"{json_out}\"\r".encode()
        
        joern_process.stdin.write(import_cpg_cmd)
        print(joern_process.stdout.readline().decode())
        joern_process.stdin.write(run_script_cmd)
        print(joern_process.stdout.readline().decode())
        joern_process.stdin.write("delete\r".encode())
        print(joern_process.stdout.readline().decode())

        try:
            out, err = joern_process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            joern_process.kill()
            out, err = joern_process.communicate()
            print("Timeout")
        print(out)
        print(err)
    
    joern_process.kill()
            

def joern_create(joern_path, in_path, out_path, cpg_files, use_cache=False):    
    json_files = []
    for cpg_file in cpg_files:
        json_file_name = f"{cpg_file.split('.')[0]}.json"
        json_files.append(json_file_name)

    # for cpg_file in cpg_files:
    #     process_joern_file(joern_path, in_path, out_path, cpg_file)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for cpg_file in cpg_files:
            if use_cache:
                json_file_name = f"{cpg_file.split('.')[0]}.json"
                if os.path.exists(out_path+json_file_name):
                    continue
            futures.append(executor.submit(process_joern_file, joern_path, in_path, out_path, cpg_file))
        print(f"len(futures): {len(futures)}")
        for future in futures:
            future.result()
            
    return json_files


def json_process(in_path, json_file):
    if os.path.exists(in_path+json_file):
        with open(in_path+json_file, encoding="utf-8") as jf:
            cpg_string = jf.read()
            cpg_string = re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", '', cpg_string)#去掉一些无用的字符串
            cpg_json = json.loads(cpg_string)
            container = [graph_indexing(graph) for graph in cpg_json["functions"] if graph["file"] != "N/A"]
            return container
    return None

'''
def generate(dataset, funcs_path):
    dataset_size = len(dataset)
    print("Size: ", dataset_size)
    graphs = funcs_to_graphs(funcs_path[2:])
    print(f"Processing CPG.")
    container = [graph_indexing(graph) for graph in graphs["functions"] if graph["file"] != "N/A"]
    graph_dataset = data.create_with_index(container, ["Index", "cpg"])
    print(f"Dataset processed.")

    return data.inner_join_by_index(dataset, graph_dataset)
'''

# client = CPGClientWrapper()
# client.create_cpg("../../data/joern/")
# joern_parse("../../joern/joern-cli/", "../../data/joern/", "../../joern/joern-cli/", "gen_test")
# print(funcs_to_graphs("/data/joern/"))
"""
while True:
    raw = input("query: ")
    response = client.query(raw)
    print(response)
"""