GPU=$1

BEST_INITIAL_RANKER=$(ls initial_ranking_model/F1* -d | sort -r | head -n 1)
echo "Found best initial ranker ${BEST_INITIAL_RANKER}"
DREX_EXPLANATION_POLICY=$(find D_REX/ -name final.*expl_model)
echo "Found best explanation policy ${DREX_EXPLANATION_POLICY}"
DREX_RERANKER=$(find D_REX/ -name final.*reranker_model)
echo "Found best reranker ${DREX_RERANKER}"

THRESHOLD1=$(echo $DREX_RERANKER | grep -o "T1.....")
THRESHOLD1=${THRESHOLD1: -2}
THRESHOLD2=$(echo $DREX_RERANKER | grep -o "T2.....")
THRESHOLD2=${THRESHOLD2: -2}

CUDA_VISIBLE_DEVICES=$GPU python3 test_drex.py \
    --relation_extraction_ranker_path=$BEST_INITIAL_RANKER \
    --explanation_policy_path=$DREX_EXPLANATION_POLICY \
    --relation_extraction_reranker_path=$DREX_RERANKER \
    --threshold1=$THRESHOLD1 \
    --threshold2=$THRESHOLD2 \
    --data_split=test