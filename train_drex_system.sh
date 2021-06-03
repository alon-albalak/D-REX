#find final file: VARNAME=$(find model_path -name final*)
# get best dev f1: VARNAME=$(ls model_path/F1* -d | sort -r | head -n 1)

GPU=$1

# pre-train initial relation extraction model
MODEL_PATH=initial_ranking_model
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

INITIAL_RANKER=$(find $MODEL_PATH -name final*)
echo "Found final initial ranker ${INITIAL_RANKER}"

# pre-train explanation policy
MODEL_PATH=explanation_policy
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

EXPL_POLICY=$(find $MODEL_PATH -name final*)
echo "Found final explanation policy ${EXPL_POLICY}"

# predict explanations for all data partitions using best explanation policy
CUDA_VISIBLE_DEVICES=$GPU python3 test_explanation_policy.py \
    --data_split=train \
    --model_class=explanation_policy_roberta \
    --model_name_or_path=$EXPL_POLICY \
    --base_model=roberta-base \
    --keep_case \
    --fp16 \
    --output_dir=$EXPL_POLICY \
    --explanation_policy_pretraining \
    --predict_trigger_for_unlabelled \
    --gpu_batch_size=30

CUDA_VISIBLE_DEVICES=$GPU python3 test_explanation_policy.py \
    --data_split=dev \
    --model_class=explanation_policy_roberta \
    --model_name_or_path=$EXPL_POLICY \
    --base_model=roberta-base \
    --keep_case \
    --fp16 \
    --output_dir=$EXPL_POLICY \
    --explanation_policy_pretraining \
    --predict_trigger_for_unlabelled \
    --gpu_batch_size=30

CUDA_VISIBLE_DEVICES=$GPU python3 test_explanation_policy.py \
    --data_split=test \
    --model_class=explanation_policy_roberta \
    --model_name_or_path=$EXPL_POLICY \
    --base_model=roberta-base \
    --keep_case \
    --fp16 \
    --output_dir=$EXPL_POLICY \
    --explanation_policy_pretraining \
    --predict_trigger_for_unlabelled \
    --gpu_batch_size=30

MODEL_PATH=reranking_model
mkdir $MODEL_PATH
CUDA_VISIBLE_DEVICES=$GPU python3 train_relation_extraction_model.py \
    --model_class=relation_extraction_roberta \
    --model_name_or_path=roberta-base \
    --base_model=roberta-base \
    --effective_batch_size=30 \
    --gpu_batch_size=30 \
    --fp16 \
    --output_dir=$MODEL_PATH \
    --relation_extraction_conditioned_on_explanations \
    --use_predicted_explanations \
    --predicted_explanation_path=$EXPL_POLICY \
    > $MODEL_PATH/train_outputs.log

RERANKER=$(find $MODEL_PATH -name final*)
echo "Found final reranker ${RERANKER}"

MODEL_PATH=D_REX_system
mkdir $MODEL_PATH
CUDA_VISIBLE_DEVICES=$GPU python3 train_drex.py \
    --relation_extraction_ranker_path=$INITIAL_RANKER \
    --explanation_policy_path=$EXPL_POLICY \
    --relation_extraction_reranker_path=$RERANKER \
    --effective_batch_size=30 \
    --gpu_batch_size=1 \
    --fp16 \
    --output_dir=$MODEL_PATH \
    --reranking_reward \
    --supervised_expl_loss \
    --leave_one_out_loss_reranker \
    > $MODEL_PATH/train_outputs.log
