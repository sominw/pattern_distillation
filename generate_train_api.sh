# togetherai run models, Mixtral and Llama-3-70B on CNN, Rotten, Cochrane
# python3 generate_train.py --dataset cnn_dailymail --model_id meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --together_api true
# python3 generate_train.py --dataset cnn_dailymail --model_id mistralai/Mixtral-8x7B-Instruct-v0.1 --together_api true

# python3 generate_train.py --dataset rotten_tomatoes --model_id meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --together_api true
# python3 generate_train.py --dataset rotten_tomatoes --model_id mistralai/Mixtral-8x7B-Instruct-v0.1 --together_api true

# python3 generate_train.py --dataset cochrane --model_id meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --together_api true
# python3 generate_train.py --dataset cochrane --model_id mistralai/Mixtral-8x7B-Instruct-v0.1 --together_api true

python3 generate_train.py --dataset pubmedsum --model_id meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --together_api true
python3 generate_train.py --dataset pubmedsum --model_id mistralai/Mixtral-8x7B-Instruct-v0.1 --together_api true