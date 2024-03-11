# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC")
# # declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC" "KICH" "COAD" "READ")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python main_tumorDetection.py --cancer $cancer \
#                   --model_path="./models_tumorDetection/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-6 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=150 \
#                   --batch_size=16 \
#                   --acc_grad=3 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=1 \
#                   --split_ratio=1 \
#                   --device="cuda"
#     done
#     done


# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python main_tumorDetection.py --cancer $cancer \
#                   --model_path="./models_tumorDetection/" \
#                   --weight_path="" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-6 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=150 \
#                   --batch_size=16 \
#                   --acc_grad=2 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.9 \
#                   --split_ratio=1 \
#                   --fair_lambda=1 \
#                   --constraint="MMF" \
#                   --reweight \
#                   --selection="avgEOpp" \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC")
# PARTITION=(2)
# CURR=(0 1 2 3)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do for curr in ${CURR[@]};
# do python main_tumorDetection.py --cancer $cancer \
#                   --model_path="./models_tumorDetection/" \
#                   --partition=$partition \
#                   --curr_fold=$curr \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-5 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=300 \
#                   --batch_size=16 \
#                   --acc_grad=2 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.8 \
#                   --split_ratio=1 \
#                   --device="cuda"
# done
# done
# done

# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC")
# PARTITION=(2)
# CURR=(0 1 2 3)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do for curr in ${CURR[@]};
# do python main_tumorDetection.py --cancer $cancer \
#                   --model_path="./models_tumorDetection/" \
#                   --partition=$partition \
#                   --curr_fold=$curr \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-6 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=300 \
#                   --batch_size=16 \
#                   --acc_grad=2 \
#                   --scheduler_step=15 \
#                   --scheduler_gamma=0.8 \
#                   --split_ratio=1 \
#                   --fair_lambda=1 \
#                   --constraint="MMF" \
#                   --reweight \
#                   --selection="avgEOpp" \
#                   --device="cuda"
# done
# done
# done


# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC")
# # declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC" "KICH" "COAD" "READ")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference_tumorDetection.py --cancer $cancer \
#                   --model_path="./models_tumorDetection/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --seed=0 \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference_tumorDetection.py --cancer $cancer \
#                   --model_path="./models_tumorDetection/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --seed=0 \
#                   --reweight \
#                   --device="cuda"
#     done
#     done

declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC")
PARTITION=(2)
SENSITIVE='{"race": ["white", "black or african american"]}'

for cancer in "${CANCER[@]}";
do for partition in ${PARTITION[@]};
do python inference_tumorDetection.py --cancer $cancer \
                  --model_path="./models_tumorDetection/" \
                  --partition=$partition \
                  --fair_attr="$SENSITIVE" \
                  --seed=0 \
                  --device="cuda"
    done
    done


declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC")
PARTITION=(2)
SENSITIVE='{"race": ["white", "black or african american"]}'

for cancer in "${CANCER[@]}";
do for partition in ${PARTITION[@]};
do python inference_tumorDetection.py --cancer $cancer \
                  --model_path="./models_tumorDetection/" \
                  --partition=$partition \
                  --fair_attr="$SENSITIVE" \
                  --seed=0 \
                  --reweight \
                  --device="cuda"
    done
    done
