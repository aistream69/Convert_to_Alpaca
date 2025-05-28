import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datasets import load_from_disk

parser = argparse.ArgumentParser(description='')
parser.add_argument('--input', default='', type=str, metavar='PATH', required=True,
                    help='Input file')
parser.add_argument('--output', default=None, type=str, metavar='PATH',
                    help='Output file')
parser.add_argument('--output_type', default='sft', type=str, metavar='TYPE',
                    help='Output data type, sft or pt')
parser.add_argument('--display', action='store_true',
                    help='Display the message of the input file')
parser.add_argument('--convert', action='store_true',
                    help='Convert training data')
parser.add_argument('--split', default=None, type=int, metavar='N',
                    help='Split N pieces of data from the input file')
parser.add_argument('--split_mode', default=None, type=str, metavar='MODE',
                    help='Split N pieces of data with MODE, example: SORT_BY_LEN')
parser.add_argument('--key_map', nargs='+', default=None, type=str,
        help='A list of key map for input and output, example: Query:instruction input:input Answer:output')
args = parser.parse_args()

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json_file(json_data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

def load_parquet_file(file_path):
    df = pd.read_parquet(file_path)
    return df

def load_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"parse json failed: {e}")
    return data

def display_json_message(data):
    print("[Info] keys:", data[0].keys())
    for l in data:
        valid = 0
        for key in l.keys():
            len_ = len(l[key]) if hasattr(l[key], '__len__') else 0
            print(f"[Info] '{key}':", l[key], f"len:{len_}")
            if len_ > 0:
                valid += 1
        if valid > 0:
            break
    print("Num examples:", len(data))

def display_dataframe_message(df):
    print("[Info] keys:", df.keys())
    for i in range(len(df)):
        valid = 0
        for key in df.keys():
            len_ = len(df[key].iloc[i]) if hasattr(df[key].iloc[i], '__len__') else 0
            print(f"[Info] '{key}':", df[key].iloc[i], f" len:{len_}")
            if len_ > 0:
                valid += 1
        if valid > 0:
            break
    print("Num examples:", len(df))

def display_file_message():
    suffix = os.path.splitext(args.input)[1]
    if suffix == '.json':
        data = load_json_file(args.input)
        display_json_message(data)
    elif suffix == '.parquet':
        df = load_parquet_file(args.input)
        display_dataframe_message(df)
    elif suffix == '.csv':
        df = pd.read_csv(args.input)
        display_dataframe_message(df)
    elif suffix == '.jsonl':
        data = load_jsonl_file(args.input)
        display_json_message(data)
    elif suffix == '.arrow':
        directory = os.path.dirname(args.input)
        dataset = load_from_disk(directory)
        df = dataset.to_pandas()
        display_dataframe_message(df)
    else:
        print(f"[Warning] unsupport suffix: {suffix}")

def split_file():
    if args.output == None:
        print(f"[Warning] Please set the output PATH first")
        return

    print(f"[Info] Split {args.split} entries into {args.output} ...")

    suffix = os.path.splitext(args.input)[1]
    if suffix == '.json':
        data = load_json_file(args.input)
        if args.split_mode == None:
            data = data[:args.split]
            save_json_file(data, args.output)
        elif args.split_mode == 'SORT_BY_LEN':
            for entry in data:
                entry['length'] = len(entry['instruction']) + len(entry['input']) + len(entry['output'])
            sorted_data = sorted(data, key=lambda x: x['length'], reverse=True)
            top_n_data = sorted_data[:args.split]
            for entry in top_n_data:
                entry.pop('length')
            save_json_file(top_n_data, args.output)
        else:
            print(f"[Warning] unsupport split mode:{args.split_mode}")
    else:
        print(f"[Warning] unsupport suffix:{suffix}, please convert to alpaca json first")

    print(f"[Info] Split ok")

def convert_dataframe(df, result, key_map):
    input_keys = []
    output_keys = []
    for m in key_map:
        m = m.split(':')
        input_keys.append(m[0])
        output_keys.append(m[1])
    for i in range(len(df)):
        data = {}
        valid = 0
        for index, i_key in enumerate(input_keys):
            o_key = output_keys[index]
            val = df[i_key].iloc[i]
            if isinstance(val, np.ndarray) or isinstance(val, np.int64):
                val = np.array2string(val)
            if i_key in df.keys() and len(val) > 0:
                data[o_key] = val
                valid += 1
            else:
                data[o_key] = ''
        if valid > 0:
            result.append(data)

def convert_json(data_in, result, key_map):
    input_keys = []
    output_keys = []
    for m in key_map:
        m = m.split(':')
        input_keys.append(m[0])
        output_keys.append(m[1])
    for l in data_in:
        data = {}
        valid = 0
        for index, i_key in enumerate(input_keys):
            o_key = output_keys[index]
            if i_key in l.keys() and len(l[i_key]) > 0:
                data[o_key] = l[i_key]
                valid += 1
            else:
                data[o_key] = ''
        if valid > 0:
            result.append(data)
    print("Num examples:", len(data_in))

def convert_data():
    if args.output == None:
        print(f"[Warning] Please set the output PATH first")
        return

    print(f"[Info] Convert {args.output_type}...")

    result = []
    key_map = args.key_map
    suffix = os.path.splitext(args.input)[1]
    if suffix == '.parquet':
        df = load_parquet_file(args.input)
        if args.output_type == 'pt':
            if key_map == None:
                key_map = ['text:text']
                print("[Info] use default pt key map:", key_map)
            else:
                print("[Info] pt key map:", key_map)
            convert_dataframe(df, result, key_map)
        elif args.output_type == 'sft':
            if key_map == None:
                key_map = ['instruction:instruction', 'input:input', 'output:output']
                print("[Info] use default sft key map:", key_map)
            else:
                print("[Info] sft key map:", key_map)
            convert_dataframe(df, result, key_map)
        else:
            print("[Warning] unsupport output type:", args.output_type)
    elif suffix == '.csv':
        df = pd.read_csv(args.input)
        if args.output_type == 'sft':
            if key_map == None:
                key_map = ['instruction:instruction', 'input:input', 'output:output']
                print("[Info] use default sft key map:", key_map)
            else:
                print("[Info] sft key map:", key_map)
            convert_dataframe(df, result, key_map)
        else:
            print("[Warning] with csv input, unsupport output type:", args.output_type)
    elif suffix == '.jsonl' or suffix == '.json':
        if suffix == '.jsonl':
            data = load_jsonl_file(args.input)
        else:
            data = load_json_file(args.input)
        if args.output_type == 'pt':
            if key_map == None:
                key_map = ['text:text']
                print("[Info] use default pt key map:", key_map)
            else:
                print("[Info] pt key map:", key_map)
            convert_json(data, result, key_map)
        elif args.output_type == 'sft':
            if key_map == None:
                key_map = ['instruction:instruction', 'input:input', 'output:output']
                print("[Info] use default sft key map:", key_map)
            else:
                print("[Info] sft key map:", key_map)
            convert_json(data, result, key_map)
        else:
            print("[Warning] unsupport output type:", args.output_type)
    elif suffix == '.arrow':
        directory = os.path.dirname(args.input)
        dataset = load_from_disk(directory)
        df = dataset.to_pandas()
        if args.output_type == 'pt':
            if key_map == None:
                key_map = ['text:text']
                print("[Info] use default pt key map:", key_map)
            else:
                print("[Info] pt key map:", key_map)
            convert_dataframe(df, result, key_map)
        elif args.output_type == 'sft':
            if key_map == None:
                key_map = ['instruction:instruction', 'input:input', 'output:output']
                print("[Info] use default sft key map:", key_map)
            else:
                print("[Info] sft key map:", key_map)
            convert_dataframe(df, result, key_map)
        else:
            print("[Warning] unsupport output type:", args.output_type)
    else:
        print(f"[Warning] unsupport suffix: {suffix}")
    if len(result) == 0:
        print(f"[Warning] result is empty, please check the key_map")
    save_json_file(result, args.output)

    print(f"[Info] Convert ok, len:{len(result)}")

def main():
    if args.display:
        display_file_message()
    elif args.split:
        split_file()
    elif args.convert:
        convert_data()

if __name__ == '__main__':
    main()

