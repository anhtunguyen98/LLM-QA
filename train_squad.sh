python finetuning_llama.py \
                --model_name meta-llama/Llama-2-7b-chat-hf \
                --dataset_name data/squad_v2 \
                --bf16 \
                --max_seq_length 2048 \
                --per_device_train_batch_size 4 \
                --gradient_accumulation_steps 4 \
                --max_steps 20000 \
                --merge_and_push \
                --save_steps 1000 \
                --learning_rate=2e-7