############## Cancer Classification ###############
# declare -a CANCER=("COAD READ")
# # declare -a CANCER=("LUAD LUSC" "KIRP KIRC KICH" "KIRP KIRC" "KIRP KICH" "KIRC KICH" "COAD READ" "GBM LGG")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference.py --cancer $cancer \
#                   --model_path="./models_CHIEF/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=1 \
#                   --seed=0 \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("LUAD LUSC")
# # declare -a CANCER=("LUAD LUSC" "KIRP KIRC KICH" "KIRP KIRC" "KIRP KICH" "KIRC KICH" "COAD READ" "GBM LGG")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference.py --cancer $cancer \
#                   --model_path="./models_CHIEF/" \
#                   --partition=$partition \
#                   --weight_path="" \
#                   --fair_attr="$SENSITIVE" \
#                   --task=1 \
#                   --seed=0 \
#                   --reweight \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("LUAD LUSC")
# # declare -a CANCER=("LUAD LUSC" "KIRP KIRC KICH" "KIRP KIRC" "KIRP KICH" "KIRC KICH" "COAD READ" "GBM LGG")
# PARTITION=(2)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference.py --cancer $cancer \
#                   --model_path="./models_CHIEF/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=1 \
#                   --seed=0 \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("LUAD LUSC")
# # declare -a CANCER=("LUAD LUSC" "KIRP KIRC KICH" "KIRP KIRC" "KIRP KICH" "KIRC KICH" "COAD READ" "GBM LGG")
# PARTITION=(2)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference.py --cancer $cancer \
#                   --model_path="./models_CHIEF/" \
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

declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC")
PARTITION=(1)
SENSITIVE='{"race": ["white", "black or african american"]}'

for cancer in "${CANCER[@]}";
do for partition in ${PARTITION[@]};
do python inference.py --cancer $cancer \
                  --model_path="./models_CHIEF/" \
                  --partition=$partition \
                  --fair_attr="$SENSITIVE" \
                  --task=2 \
                  --seed=0 \
                  --device="cuda"
    done
    done

declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC")
PARTITION=(1)
SENSITIVE='{"race": ["white", "black or african american"]}'

for cancer in "${CANCER[@]}";
do for partition in ${PARTITION[@]};
do python inference.py --cancer $cancer \
                  --model_path="./models_CHIEF/" \
                  --partition=$partition \
                  --fair_attr="$SENSITIVE" \
                  --task=2 \
                  --seed=0 \
                  --reweight \
                  --device="cuda"
    done
    done

declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC")
PARTITION=(2)
SENSITIVE='{"race": ["white", "black or african american"]}'

for cancer in "${CANCER[@]}";
do for partition in ${PARTITION[@]};
do python inference.py --cancer $cancer \
                  --model_path="./models_CHIEF/" \
                  --partition=$partition \
                  --fair_attr="$SENSITIVE" \
                  --task=2 \
                  --seed=0 \
                  --device="cuda"
    done
    done

declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC")
PARTITION=(2)
SENSITIVE='{"race": ["white", "black or african american"]}'

for cancer in "${CANCER[@]}";
do for partition in ${PARTITION[@]};
do python inference.py --cancer $cancer \
                  --model_path="./models_CHIEF/" \
                  --partition=$partition \
                  --fair_attr="$SENSITIVE" \
                  --task=2 \
                  --seed=0 \
                  --reweight \
                  --device="cuda"
    done
    done


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

# ############## Genetic Mutation Classification ###############

# declare -a CANCER=("lusc" "kirp" "kirc" "kich" "coadread" "lgg" "gbm")
# PARTITION=(1)
# SENSITIVE='{"Sex": ["Female", "Male"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference_genetic.py --cancer $cancer \
#                   --model_path="./models_CHIEF/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=4 \
#                   --seed=0 \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("luad" "lusc" "kirp" "kirc" "kich" "coad" "read" "gbm" "lgg")
# PARTITION=(1)
# SENSITIVE='{"Sex": ["Female", "Male"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference_genetic.py --cancer $cancer \
#                   --model_path="./models_pan_cancer/" \
#                   --weight_path="" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=4 \
#                   --seed=0 \
#                   --reweight \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("luad" "lusc" "kirp" "kirc")
# PARTITION=(2)
# SENSITIVE='{"Sex": ["Female", "Male"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference_genetic.py --cancer $cancer \
#                   --model_path="./models_pan_cancer/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=4 \
#                   --seed=0 \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("luad" "lusc" "kirp" "kirc")
# PARTITION=(2)
# SENSITIVE='{"Sex": ["Female", "Male"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference_genetic.py --cancer $cancer \
#                   --model_path="./models_pan_cancer/" \
#                   --weight_path="" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=4 \
#                   --seed=0 \
#                   --reweight \
#                   --device="cuda"
#     done
#     done