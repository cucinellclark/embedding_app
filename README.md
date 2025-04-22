# BV-BRC Module

## Overview

This repository holds a BV-BRC module.

## About this module

This module is a component of the BV-BRC build system. It is designed to fit into the
`dev_container` infrastructure which manages development and production deployment of
the components of the BV-BRC. More documentation is available [here](https://github.com/BV-BRC/dev_container/tree/master/README.md).

This is hard coded to work with the following response schema:
<pre>
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
</pre>

Outputs the embeddings in a jsonl file like so:
<pre>
{
    "index": 0,
    "object": "embedding",
    "source": source file,
    "doc_id": unique generated id linking back to full document,
    "chunk_index": index of the chunk of the full document,
    "chunk_text": the embedding text of this chunk (could be the full document)
    "embedding": [...]
}
</pre>

Will accept two formats for input: a folder or a jsonl file

Below is the expected format for the file input, along with each file in the folder input option
<pre>
{"doc_id": "unique_id_1", "text": "This is the document text I would like to embed...", "metadata": [ ... ]}
{"doc_id": "unique_id_2", "text": "This is the document text I would like to embed that is different...", "metadata": [ ... ]}
</pre>
