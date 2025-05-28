# Convert_to_Alpaca
Convert the Hugging Face datasets to Alpaca format, which can be used for LLaMA-Factory.
****

#### Supported input formats
- Parquet
- Arrow
- JSON
- JSON Lines (JSONL)
- CSV

#### Supported functions
    $ python convert.py -h
    options:
        -h, --help            show this help message and exit
        --input PATH          Input file
        --output PATH         Output file
        --output_type TYPE    Output data type, sft or pt
        --display             Display the message of the input file
        --convert             Convert training data
        --split N             Split N pieces of data from the input file
        --split_mode MODE     Split N pieces of data with MODE, example: SORT_BY_LEN
        --key_map KEY_MAP [KEY_MAP ...]
                              A list of key map for input and output, example: Query:instruction input:input Answer:output

#### Examples
    python convert.py --input test.parquet --display
    python convert.py --input test.arrow --display
    python convert.py --input test.csv --display
    python convert.py --input test.json --display
    python convert.py --input test.jsonl --display

    python convert.py --input test.parquet --output alpaca.json --convert
    python convert.py --input test.parquet --output alpaca.json --convert --output_type pt
    python convert.py --input test.parquet --output alpaca.json --convert --key_map problem:instruction input:input solution:output

    python convert.py --input alpaca.json --output slip.json --split 10

