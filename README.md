# BV-BRC Module

## Overview

This repository holds a BV-BRC module.

## About this module

This module is a component of the BV-BRC build system. It is designed to fit into the
`dev_container` infrastructure which manages development and production deployment of
the components of the BV-BRC. More documentation is available [here](https://github.com/BV-BRC/dev_container/tree/master/README.md).

This is hard coded to work with the following response schema:

{
    'id': vllm-server-id,
    'object': 'list',
    'created': created_date
    'model': model_name,
    'data': [
        {
            'index': 0,
            'object': 'embedding',
            'embedding': [...],
            'usage': { tokenization_data }
        }
    ]
}

Outputs the embeddings in a jsonl file like so:
{
    "doc_id": "unique_id",
    "embedding": [...]
}

Will accept two formats for input: a folder or a jsonl file

The folder should contain json files with the following format:
{
    "doc_id": "unique_id",
    "text": "This is the document text I would like to embed..."
    "metadata": [
        ...
    ]
}

The file should be a jsonl document where each line is a json object. Example below:
{"doc_id": "unique_id_1", "text": "This is the document text I would like to embed...", "metadata": [ ... ]}
{"doc_id": "unique_id_2", "text": "This is the document text I would like to embed that is different...", "metadata": [ ... ]}
