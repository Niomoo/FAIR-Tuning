# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRC")
# # declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC" "KICH" "COAD")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python main_survivalPrediction.py --cancer $cancer \
#                   --model_path="./models_survivalPrediction/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-6 \
#                   --dropout=0.3 \
#                   --seed=5 \
#                   --epochs=200 \
#                   --batch_size=16 \
#                   --acc_grad=1 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=1 \
#                   --split_ratio=1 \
#                   --device="cuda"
#     done
#     done


# declare -a CANCER=("LUAD" "LUSC" "KIRC")
# declare -a CANCER=("BRCA")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python main_survivalPrediction.py --cancer $cancer \
#                   --model_path="./models_survivalPrediction/" \
#                   --weight_path="28" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-6 \
#                   --dropout=0.3 \
#                   --seed=5 \
#                   --epochs=100 \
#                   --batch_size=16 \
#                   --acc_grad=1 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.9 \
#                   --split_ratio=1 \
#                   --fair_lambda=1 \
#                   --reweight \
#                   --selection="c_diff" \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("LUAD")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python main_survivalPrediction.py --cancer $cancer \
#                   --model_path="./models_survivalPrediction/" \
#                   --weight_path="27" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-5 \
#                   --dropout=0.3 \
#                   --seed=5 \
#                   --epochs=100 \
#                   --batch_size=16 \
#                   --acc_grad=1 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.9 \
#                   --split_ratio=1 \
#                   --fair_lambda=1 \
#                   --reweight \
#                   --selection="c_diff" \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("LUSC")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python main_survivalPrediction.py --cancer $cancer \
#                   --model_path="./models_survivalPrediction/" \
#                   --weight_path="26" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-6 \
#                   --dropout=0.3 \
#                   --seed=5 \
#                   --epochs=100 \
#                   --batch_size=16 \
#                   --acc_grad=1 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.9 \
#                   --split_ratio=1 \
#                   --fair_lambda=1 \
#                   --reweight \
#                   --selection="c_diff" \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("KIRC")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python main_survivalPrediction.py --cancer $cancer \
#                   --model_path="./models_survivalPrediction/" \
#                   --weight_path="26" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-6 \
#                   --dropout=0.3 \
#                   --seed=5 \
#                   --epochs=100 \
#                   --batch_size=16 \
#                   --acc_grad=1 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.9 \
#                   --split_ratio=1 \
#                   --fair_lambda=1 \
#                   --reweight \
#                   --selection="c_diff" \
#                   --device="cuda"
#     done
#     done

declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRC")
PARTITION=(2)
CURR=(0 1 2 3)
SENSITIVE='{"race": ["white", "black or african american"]}'

for cancer in "${CANCER[@]}";
do for partition in ${PARTITION[@]};
do for curr in ${CURR[@]};
do python main_survivalPrediction.py --cancer $cancer \
                  --model_path="./models_survivalPrediction/" \
                  --partition=$partition \
                  --curr_fold=$curr \
                  --fair_attr="$SENSITIVE" \
                  --lr=1e-5 \
                  --dropout=0.2 \
                  --seed=3 \
                  --epochs=200 \
                  --batch_size=16 \
                  --acc_grad=3 \
                  --scheduler_step=10 \
                  --scheduler_gamma=0.8 \
                  --split_ratio=1 \
                  --device="cuda"
done
done
done

# declare -a CANCER=("BRCA" "LUSC" "KIRC")
# # declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRC")
# PARTITION=(2)
# CURR=(0 1 2 3)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do for curr in ${CURR[@]};
# do python main_survivalPrediction.py --cancer $cancer \
#                   --model_path="./models_survivalPrediction/" \
#                   --partition=$partition \
#                   --curr_fold=$curr \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-5 \
#                   --dropout=0.3 \
#                   --seed=3 \
#                   --epochs=150 \
#                   --batch_size=16 \
#                   --acc_grad=2 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.8 \
#                   --split_ratio=1 \
#                   --fair_lambda=1 \
#                   --reweight \
#                   --selection="c_diff" \
#                   --device="cuda"
# done
# done
# done


# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRC")
# # declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC" "KICH" "COAD")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference_survivalPrediction.py --cancer $cancer \
#                   --model_path="./models_survivalPrediction/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --seed=5 \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRC")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference_survivalPrediction.py --cancer $cancer \
#                   --model_path="./models_survivalPrediction/" \
#                   --weight_path="" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --seed=5 \
#                   --reweight \
#                   --device="cuda"
#     done
#     done

declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRC")
PARTITION=(2)
SENSITIVE='{"race": ["white", "black or african american"]}'

for cancer in "${CANCER[@]}";
do for partition in ${PARTITION[@]};
do python inference_survivalPrediction.py --cancer $cancer \
                  --model_path="./models_survivalPrediction/" \
                  --partition=$partition \
                  --fair_attr="$SENSITIVE" \
                  --seed=3 \
                  --device="cuda"
    done
    done

# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRC")
# PARTITION=(2)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference_survivalPrediction.py --cancer $cancer \
#                   --model_path="./models_survivalPrediction/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --seed=3 \
#                   --reweight \
#                   --device="cuda"
#     done
#     done
