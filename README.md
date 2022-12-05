## The repo contains the code to train and evaluate a system which extracts relations and explanations from dialogue.
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/d-rex-dialogue-relation-extraction-with/dialog-relation-extraction-on-dialogre)](https://paperswithcode.com/sota/dialog-relation-extraction-on-dialogre?p=d-rex-dialogue-relation-extraction-with)

#### How do I cite D-REX?
Please cite [the ConvAI paper](https://aclanthology.org/2022.nlp4convai-1.4/)
```
@inproceedings{albalak-etal-2022-rex,
    title = "{D}-{REX}: Dialogue Relation Extraction with Explanations",
    author = "Albalak, Alon  and
      Embar, Varun  and
      Tuan, Yi-Lin  and
      Getoor, Lise  and
      Wang, William Yang",
    booktitle = "Proceedings of the 4th Workshop on NLP for Conversational AI",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.nlp4convai-1.4",
    doi = "10.18653/v1/2022.nlp4convai-1.4",
    pages = "34--46",
    abstract = "Existing research studies on cross-sentence relation extraction in long-form multi-party conversations aim to improve relation extraction without considering the explainability of such methods. This work addresses that gap by focusing on extracting explanations that indicate that a relation exists while using only partially labeled explanations. We propose our model-agnostic framework, D-REX, a policy-guided semi-supervised algorithm that optimizes for explanation quality and relation extraction simultaneously. We frame relation extraction as a re-ranking task and include relation- and entity-specific explanations as an intermediate step of the inference process. We find that human annotators are 4.2 times more likely to prefer D-REX{'}s explanations over a joint relation extraction and explanation model. Finally, our evaluations show that D-REX is simple yet effective and improves relation extraction performance of strong baseline models by 1.2-4.7{\%}.",
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
