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

BEST_INITIAL_RANKER=$(ls $MODEL_PATH/F1* -d | sort -r | head -n 1)
echo "Found best initial ranker ${BEST_INITIAL_RANKER}"

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

BEST_EXPL_POLICY=$(find $MODEL_PATH -name final*)
echo "Found best explanation policy ${BEST_EXPL_POLICY}"

# predict explanations for all data partitions using best explanation policy
CUDA_VISIBLE_DEVICES=$GPU python3 test_explanation_policy.py \
    --data_split=train \
    --model_class=explanation_policy_roberta \
    --model_name_or_path=$BEST_EXPL_POLICY \
    --base_model=roberta-base \
    --keep_case \
    --fp16 \
    --output_dir=$BEST_EXPL_POLICY \
    --explanation_policy_pretraining \
    --predict_trigger_for_unlabelled \
    --gpu_batch_size=30

CUDA_VISIBLE_DEVICES=$GPU python3 test_explanation_policy.py \
    --data_split=dev \
    --model_class=explanation_policy_roberta \
    --model_name_or_path=$BEST_EXPL_POLICY \
    --base_model=roberta-base \
    --keep_case \
    --fp16 \
    --output_dir=$BEST_EXPL_POLICY \
    --explanation_policy_pretraining \
    --predict_trigger_for_unlabelled \
    --gpu_batch_size=30

CUDA_VISIBLE_DEVICES=$GPU python3 test_explanation_policy.py \
    --data_split=test \
    --model_class=explanation_policy_roberta \
    --model_name_or_path=$BEST_EXPL_POLICY \
    --base_model=roberta-base \
    --keep_case \
    --fp16 \
    --output_dir=$BEST_EXPL_POLICY \
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
    --predicted_explanation_path=$BEST_EXPL_POLICY \
    > $MODEL_PATH/train_outputs.log

BEST_RERANKER=$(ls $MODEL_PATH/F1* -d | sort -r | head -n 1)
echo "Found best reranker ${BEST_RERANKER}"

MODEL_PATH=D_REX_system
mkdir $MODEL_PATH
CUDA_VISIBLE_DEVICES=$GPU python3 train_drex.py \
    --relation_extraction_ranker_path=$BEST_INITIAL_RANKER \
    --explanation_policy_path=$BEST_EXPL_POLICY \
    --relation_extraction_reranker_path=$BEST_RERANKER \
    --effective_batch_size=30 \
    --gpu_batch_size=1 \
    --fp16 \
    --output_dir=$MODEL_PATH \
    --reranking_reward \
    --supervised_expl_loss \
    --leave_one_out_loss_reranker \
    > $MODEL_PATH/train_outputs.log
