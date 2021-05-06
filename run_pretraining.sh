#!/bin/bash

# run bert pretraining model
python3 pretraining.py                                   \
        --bert_config_file=./bert_config.json            \
        --input_file=./data/input/eth_node_example       \
        --output_dir=./data/output                       \
        --do_train=true
