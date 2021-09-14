## The repo contains the code to train and evaluate a system which extracts relations and explanations from dialogue.

#### How do I cite D-REX?
For now, cite [the Arxiv paper](https://arxiv.org/abs/2109.05126)
```
@article{albalak2021drex,
      title={D-REX: Dialogue Relation Extraction with Explanations}, 
      author={Alon Albalak and Varun Embar and Yi-Lin Tuan and Lise Getoor and William Yang Wang},
      journal={arXiv preprint arXiv:2109.05126},
      year={2021},
}
```


### To train the full system:
```bash
GPU=0
bash train_drex_system.sh $GPU
```
Notes:
- The training script is set up to work with an NVIDIA Titan RTX (24Gb memory, mixed-precision)
- To train on a GPU with less memory, adjust the `GPU_BATCH_SIZE` parameter in `train_drex_system.sh` to match your memory limit.
- Training the full system takes ~24 hours on a single NVIDIA Titan RTX

### To test the trained system:
```bash
GPU=0
bash test_drex_system.sh $GPU
```

### To train/test individual modules:

- Relation Extraction Model -
  - Training:
    ```bash
    GPU=0
    MODEL_PATH=relation_extraction_model
    mkdir $MODEL_PATH
    CUDA_VISIBLE_DEVICES=$GPU python3 train_relation_extraction_model.py \
        --model_class=relation_extraction_roberta \
        --model_name_or_path=roberta-base \
        --base_model=roberta-base \
        --effective_batch_size=30 \
        --gpu_batch_size=30 \
        --fp16 \
        --output_dir=$MODEL_PATH \
        --relation_extraction_pretraining \
        > $MODEL_PATH/train_outputs.log
    ```
  - Testing:
    ```bash
    GPU=0
    MODEL_PATH=relation_extraction_model
    BEST_MODEL=$(ls $MODEL_PATH/F1* -d | sort -r | head -n 1)
    THRESHOLD1=$(echo $BEST_MODEL | grep -o "T1.....")
    THRESHOLD1=${THRESHOLD1: -2}
    THRESHOLD2=$(echo $BEST_MODEL | grep -o "T2.....")
    THRESHOLD2=${THRESHOLD2: -2}
    CUDA_VISIBLE_DEVICES=0 python3 test_relation_extraction_model.py \
        --model_class=relation_extraction_roberta \
        --model_name_or_path=$BEST_MODEL \
        --base_model=roberta-base \
        --relation_extraction_pretraining \
        --threshold1=$THRESHOLD1 \
        --threshold2=$THRESHOLD2 \
        --data_split=test
    ```
- Explanation Extraction Model -
  - Training:
    ```bash
    GPU=0
    MODEL_PATH=explanation_extraction_model
    mkdir $MODEL_PATH
    CUDA_VISIBLE_DEVICES=$GPU python3 train_explanation_policy.py \
        --model_class=explanation_policy_roberta \
        --model_name_or_path=roberta-base \
        --base_model=roberta-base \
        --effective_batch_size=30 \
        --gpu_batch_size=30 \
        --fp16 \
        --output_dir=$MODEL_PATH \
        --explanation_policy_pretraining \
        > $MODEL_PATH/train_outputs.log    
    ```
  - Testing:
    ```bash
    GPU=0
    MODEL_PATH=explanation_extraction_model
    BEST_MODEL=$(ls $MODEL_PATH/F1* -d | sort -r | head -n 1)
    CUDA_VISIBLE_DEVICES=$GPU python3 test_explanation_policy.py \
        --model_class=explanation_policy_roberta \
        --model_name_or_path=$BEST_MODEL \
        --base_model=roberta-base \
        --explanation_policy_pretraining \
        --data_split=test
    ```
