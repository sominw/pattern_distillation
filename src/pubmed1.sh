CUDA_VISIBLE_DEVICES="0,1" python3 clm_trainer.py --teacher=mistral7b --model=gpt2_xl --max_epochs=50 --batch_size=4 --data=pubmed
CUDA_VISIBLE_DEVICES="0,1" python3 clm_trainer.py --teacher=llama8b --model=gpt2_xl --max_epochs=50 --batch_size=4 --data=pubmed
CUDA_VISIBLE_DEVICES="0,1" python3 clm_trainer.py --teacher=gemma2_9b --model=gpt2_xl --max_epochs=50 --batch_size=4 --data=pubmed