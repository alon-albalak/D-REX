GPU=$1

INITIAL_RANKER=$(find initial_ranking_model/ -name final*)
echo "Found final initial ranker ${INITIAL_RANKER}"
DREX_EXPLANATION_POLICY=$(ls D_REX_system/F1*expl_model -d | sort -r | head -n 1)
echo "Found best explanation policy ${DREX_EXPLANATION_POLICY}"
DREX_RERANKER=$(ls D_REX_system/F1*reranker_model -d | sort -r | head -n 1)
echo "Found best reranker ${DREX_RERANKER}"

THRESHOLD1=$(echo $DREX_RERANKER | grep -o "T1.....")
THRESHOLD1=${THRESHOLD1: -2}
THRESHOLD2=$(echo $DREX_RERANKER | grep -o "T2.....")
THRESHOLD2=${THRESHOLD2: -2}

CUDA_VISIBLE_DEVICES=$GPU python3 test_drex.py \
    --relation_extraction_ranker_path=$INITIAL_RANKER \
    --explanation_policy_path=$DREX_EXPLANATION_POLICY \
    --relation_extraction_reranker_path=$DREX_RERANKER \
    --threshold1=$THRESHOLD1 \
    --threshold2=$THRESHOLD2 \
    --data_split=test