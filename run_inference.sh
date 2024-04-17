# ############## Cancer Classification ###############
# declare -a CANCER=("LUAD LUSC")
# declare -a CANCER=("LUAD LUSC" "KIRP KIRC KICH" "KIRP KIRC" "KIRP KICH" "KIRC KICH" "COAD READ" "GBM LGG")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference.py --cancer $cancer \
#                   --model_path="./models_gender/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=1 \
#                   --seed=0 \
#                   --device="cuda"
#     done
#     done

declare -a CANCER=("KIRP KICH")
# declare -a CANCER=("LUAD LUSC" "KIRP KIRC KICH" "KIRP KIRC" "KIRP KICH" "KIRC KICH" "COAD READ" "GBM LGG")
PARTITION=(1)
SENSITIVE='{"race": ["white", "black or african american"]}'

for cancer in "${CANCER[@]}";
do for partition in ${PARTITION[@]};
do python inference.py --cancer $cancer \
                  --model_path="./models_gender/" \
                  --partition=$partition \
                  --weight_path="" \
                  --fair_attr="$SENSITIVE" \
                  --task=1 \
                  --seed=0 \
                  --reweight \
                  --device="cuda"
    done
    done

# declare -a CANCER=("LUAD LUSC")
# declare -a CANCER=("LUAD LUSC" "KIRP KIRC KICH" "KIRP KIRC" "KIRP KICH" "KIRC KICH" "COAD READ" "GBM LGG")
# PARTITION=(2)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference.py --cancer $cancer \
#                   --model_path="./models_gender/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=1 \
#                   --seed=0 \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("KIRP KIRC KICH")
# declare -a CANCER=("LUAD LUSC" "KIRP KIRC KICH" "KIRP KIRC" "KIRP KICH" "KIRC KICH" "COAD READ" "GBM LGG")
# PARTITION=(2)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference.py --cancer $cancer \
#                   --model_path="./models_gender/" \
#                   --weight_path="" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=1 \
#                   --seed=0 \
#                   --reweight \
#                   --device="cuda"
#     done
#     done

# ############## Tumor Detection ###############

# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference.py --cancer $cancer \
#                   --model_path="./models/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=2 \
#                   --seed=0 \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference.py --cancer $cancer \
#                   --model_path="./models/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=2 \
#                   --seed=0 \
#                   --reweight \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC")
# PARTITION=(2)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference.py --cancer $cancer \
#                   --model_path="./models/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=2 \
#                   --seed=0 \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC")
# PARTITION=(2)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference.py --cancer $cancer \
#                   --model_path="./models/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=2 \
#                   --seed=0 \
#                   --reweight \
#                   --device="cuda"
#     done
#     done


# ############## Survival Analysis ###############

# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRC")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference.py --cancer $cancer \
#                   --model_path="./models/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=3 \
#                   --seed=5 \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRC")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference.py --cancer $cancer \
#                   --model_path="./models/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=3 \
#                   --seed=5 \
#                   --reweight \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRC")
# PARTITION=(2)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference.py --cancer $cancer \
#                   --model_path="./models/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=3 \
#                   --seed=3 \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("KIRC")
# PARTITION=(2)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference.py --cancer $cancer \
#                   --model_path="./models/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=3 \
#                   --seed=5 \
#                   --reweight \
#                   --device="cuda"
#     done
#     done
