# LLL-QA

## How to train

### Install requirements

```bash
pip install -r requirements.txt
```

### Create dataset

```bash
python preprocess_squad_data.py
python preprocess_commonsense_data.py
```

### Train model


```bash
bash train_squad.sh
```

### Evaluate model

pass the parameter `--quantize` to the script.

```bash
python test_squad.py --adapter_name=squad_final_checkpoints
```
This generates a CSV file `results/results.csv`

### Available Lora Checkpoints

Model | Huggingface hub
---|---|
Llama-2-7b-chat-squad | [checkpoint](https://huggingface.co/anhtunguyen98/llama2-squad)
Llama-2-7b-chat-commonsense | [checkpoint](https://huggingface.co/anhtunguyen98/llama-commonsense)
