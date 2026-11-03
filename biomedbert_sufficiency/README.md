# Fine-tuned sufficiency classifier

The fine-tuned `model.safetensors` weights are intentionally not stored in Git.
To run `query.py` with the classifier, obtain the weights separately and place
`model.safetensors` in this directory alongside `config.json` and the tokenizer
files. Training checkpoints and the generated vector database are also kept
locally and are not included in the repository.
