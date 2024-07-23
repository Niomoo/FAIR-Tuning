############## Cancer Classification ###############
# declare -a CANCER=("BRCA" "LUAD LUSC" "KIRP KIRC KICH" "KIRP KIRC" "KIRP KICH" "KIRC KICH" "COAD READ" "GBM LGG")
# PARTITION=(2)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python bootstrap_statistic_test.py --cancer $cancer \
#                   --model_path="./models_race/" \
#                   --weight_path="" \
#                   --reweight_path="" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=1 \
#     done
#     done

# declare -a CANCER=("LUAD LUSC" "KIRP KIRC KICH" "KIRP KIRC" "KIRP KICH" "KIRC KICH" "COAD READ" "GBM LGG")
# PARTITION=(2)
# SENSITIVE='{"gender": ["female", "male"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python bootstrap_statistic_test.py --cancer $cancer \
#                   --model_path="./models_gender/" \
#                   --weight_path="" \
#                   --reweight_path="" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=1 \
#     done
#     done

# ############## Tumor Detection ###############
# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRC")
# PARTITION=(2)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python bootstrap_statistic_test.py --cancer $cancer \
#                   --model_path="./models_race/" \
#                   --weight_path="" \
#                   --reweight_path="" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=2 \
#     done
#     done

# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRC")
# PARTITION=(2)
# SENSITIVE='{"gender": ["female", "male"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python bootstrap_statistic_test.py --cancer $cancer \
#                   --model_path="./models_gender/" \
#                   --weight_path="" \
#                   --reweight_path="" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=2 \
#     done
#     done

# ############## Survival Analysis ###############
# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRC")
# PARTITION=(2)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference_statistic.py --cancer $cancer \
#                   --model_path="./models_race/" \
#                   --weight_path="" \
#                   --reweight_path="" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=3 \
#     done
#     done

# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRC")
# PARTITION=(2)
# SENSITIVE='{"gender": ["female", "male"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference_statistic.py --cancer $cancer \
#                   --model_path="./models_gender/" \
#                   --weight_path="" \
#                   --reweight_path="" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=3 \
#     done
#     done

# ############## Genetic Mutation Classification ###############
# declare -a CANCER=("luad" "lusc")
# PARTITION=(2)
# SENSITIVE='{"Race Category": ["White", "Black or African American"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python bootstrap_statistic_test.py --cancer $cancer \
#                   --model_path="./models_race/" \
#                   --weight_path="" \
#                   --reweight_path="" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=4 \
#     done
#     done

# declare -a CANCER=("luad" "lusc")
# PARTITION=(2)
# SENSITIVE='{"Sex": ["Female", "Male"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python bootstrap_statistic_test.py --cancer $cancer \
#                   --model_path="./models_gender/" \
#                   --weight_path="" \
#                   --reweight_path="" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=4 \
#     done
#     done

############## Statistical Test ###############
# python summary_significant_test.py