
ROOT="~"

CITY_LIST=("nairobi" "london")
NUM_POINTS=20
NUM_PROBLEMS=100

PARALLEL_NUM=100


for CITY in "${CITY_LIST[@]}"; do

    python "${ROOT}/rl4co/rl4co/tasks/collect_numba_nobatch.py" \
        --project_root "${ROOT}" \
        --city "${CITY}" \
        --num_points "${NUM_POINTS}" \
        --prob_instance_num "${NUM_PROBLEMS}" \
        --mode "data"
    wait

    for ((i=0;i<PARALLEL_NUM;++i)); do
        python "${ROOT}/rl4co/rl4co/tasks/collect_numba_nobatch.py" \
            --project_root "${ROOT}" \
            --city "${CITY}" \
            --num_points "${NUM_POINTS}" \
            --prob_instance_num "${NUM_PROBLEMS}" \
            --test_idx "$i" \
            --total_idx "${PARALLEL_NUM}" \
            --mode "tsp" &
    done
    wait

    python "${ROOT}/rl4co/rl4co/tasks/collect_numba_nobatch.py" \
        --project_root "${ROOT}" \
        --city "${CITY}" \
        --num_points "${NUM_POINTS}" \
        --prob_instance_num "${NUM_PROBLEMS}" \
        --test_idx "$i" \
        --total_idx "${PARALLEL_NUM}" \
        --mode "summarize"
    wait
done
