# ############### Cancer Classification ###############

# declare -a CANCER=("KIRP KIRC KICH" "KIRP KIRC" "KIRP KICH" "KIRC KICH" "COAD READ" "GBM LGG")
# # declare -a CANCER=("LUAD LUSC" "KIRP KIRC KICH" "KIRP KIRC" "KIRP KICH" "KIRC KICH" "COAD READ" "GBM LGG")
# PARTITION=(1)
# SENSITIVE='{"gender": ["female", "male"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_gender/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=1 \
#                   --lr=1e-5 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=150 \
#                   --batch_size=16 \
#                   --acc_grad=2 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.9 \
#                   --device="cuda"
#     done
#     done


# declare -a CANCER=("KIRP KIRC KICH" "KIRP KIRC" "KIRP KICH" "KIRC KICH" "COAD READ" "GBM LGG")
declare -a CANCER=("LUAD LUSC" "KIRP KIRC KICH" "KIRP KIRC" "KIRP KICH" "KIRC KICH" "COAD READ" "GBM LGG")
PARTITION=(1)
SENSITIVE='{"gender": ["female", "male"]}'

for cancer in "${CANCER[@]}";
do for partition in ${PARTITION[@]};
do python main_base.py --cancer $cancer \
                  --model_path="./models_gender/" \
                  --weight_path="" \
                  --partition=$partition \
                  --fair_attr="$SENSITIVE" \
                  --task=1 \
                  --lr=1e-5 \
                  --dropout=0.3 \
                  --seed=0 \
                  --epochs=150 \
                  --batch_size=16 \
                  --acc_grad=1 \
                  --scheduler_step=10 \
                  --scheduler_gamma=0.9 \
                  --fair_lambda=1 \
                  --constraint="AE" \
                  --reweight \
                  --selection="EOdd" \
                  --device="cuda"
    done
    done

declare -a CANCER=("LUAD LUSC" "KIRP KIRC KICH" "KIRP KIRC" "KIRP KICH" "KIRC KICH" "COAD READ" "GBM LGG")
PARTITION=(1)
SENSITIVE='{"gender": ["female", "male"]}'

for cancer in "${CANCER[@]}";
do for partition in ${PARTITION[@]};
do python main_base.py --cancer $cancer \
                  --model_path="./models_gender/" \
                  --weight_path="" \
                  --partition=$partition \
                  --fair_attr="$SENSITIVE" \
                  --task=1 \
                  --lr=1e-5 \
                  --dropout=0.3 \
                  --seed=0 \
                  --epochs=150 \
                  --batch_size=16 \
                  --acc_grad=1 \
                  --scheduler_step=10 \
                  --scheduler_gamma=0.9 \
                  --fair_lambda=1 \
                  --constraint="MMF" \
                  --reweight \
                  --selection="EOdd" \
                  --device="cuda"
    done
    done

# declare -a CANCER=("KIRP KIRC KICH" "KIRP KIRC" "KIRP KICH" "KIRC KICH" "COAD READ" "GBM LGG")
# # declare -a CANCER=("LUAD LUSC" "KIRP KIRC KICH" "KIRP KIRC" "KIRP KICH" "KIRC KICH" "COAD READ" "GBM LGG")
# PARTITION=(2)
# CURR=(0 1 2 3)
# SENSITIVE='{"gender": ["female", "male"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do for curr in ${CURR[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_gender/" \
#                   --partition=$partition \
#                   --curr_fold=$curr \
#                   --fair_attr="$SENSITIVE" \
#                   --task=1 \
#                   --lr=1e-5 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=100 \
#                   --batch_size=16 \
#                   --acc_grad=2 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.9 \
#                   --device="cuda"
# done
# done
# done

# declare -a CANCER=("KIRP KICH" "KIRC KICH" "COAD READ" "GBM LGG")
# declare -a CANCER=("LUAD LUSC" "KIRP KIRC KICH" "KIRP KIRC" "KIRP KICH" "KIRC KICH" "COAD READ" "GBM LGG")
# PARTITION=(2)
# CURR=(0 1 2 3)
# SENSITIVE='{"gender": ["female", "male"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do for curr in ${CURR[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_gender/" \
#                   --weight_path="" \
#                   --partition=$partition \
#                   --curr_fold=$curr \
#                   --fair_attr="$SENSITIVE" \
#                   --task=1 \
#                   --lr=1e-5 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=100 \
#                   --batch_size=16 \
#                   --acc_grad=1 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.9 \
#                   --fair_lambda=1 \
#                   --constraint="MMF" \
#                   --reweight \
#                   --selection="avgEOpp" \
#                   --device="cuda"
# done
# done
# done

############### Tumor detection ###############

# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC")
# PARTITION=(1)
# SENSITIVE='{"gender": ["female", "male"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_gender/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=2 \
#                   --lr=1e-6 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=10 \
#                   --batch_size=16 \
#                   --acc_grad=3 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=1 \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC")
# PARTITION=(1)
# SENSITIVE='{"gender": ["female", "male"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_gender/" \
#                   --weight_path="" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=2 \
#                   --lr=1e-6 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=5 \
#                   --batch_size=16 \
#                   --acc_grad=2 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.9 \
#                   --fair_lambda=1 \
#                   --constraint="" \
#                   --reweight \
#                   --selection="avgEOpp" \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC")
# PARTITION=(2)
# CURR=(0 1 2 3)
# SENSITIVE='{"gender": ["female", "male"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do for curr in ${CURR[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_gender/" \
#                   --partition=$partition \
#                   --curr_fold=$curr \
#                   --fair_attr="$SENSITIVE" \
#                   --task=2 \
#                   --lr=1e-5 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=10 \
#                   --batch_size=16 \
#                   --acc_grad=2 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.9 \
#                   --device="cuda"
# done
# done
# done

# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRP" "KIRC")
# PARTITION=(2)
# CURR=(0 1 2 3)
# SENSITIVE='{"gender": ["female", "male"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do for curr in ${CURR[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_gender/" \
#                   --weight_path="" \
#                   --partition=$partition \
#                   --curr_fold=$curr \
#                   --fair_attr="$SENSITIVE" \
#                   --task=2 \
#                   --lr=1e-6 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=5 \
#                   --batch_size=16 \
#                   --acc_grad=2 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.9 \
#                   --fair_lambda=1 \
#                   --constraint="" \
#                   --reweight \
#                   --selection="avgEOpp" \
#                   --device="cuda"
# done
# done
# done

############### Survival Analysis ###############

# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRC")
# PARTITION=(1)
# SENSITIVE='{"gender": ["female", "male"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_gender/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=3 \
#                   --lr=1e-6 \
#                   --dropout=0.3 \
#                   --seed=5 \
#                   --epochs=10 \
#                   --batch_size=16 \
#                   --acc_grad=1 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=1 \
#                   --split_ratio=1 \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRC")
# PARTITION=(1)
# SENSITIVE='{"gender": ["female", "male"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_gender/" \
#                   --weight_path="" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --task=3 \
#                   --lr=1e-6 \
#                   --dropout=0.3 \
#                   --seed=5 \
#                   --epochs=5 \
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

# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRC")
# PARTITION=(2)
# CURR=(0 1 2 3)
# SENSITIVE='{"gender": ["female", "male"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do for curr in ${CURR[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_gender/" \
#                   --partition=$partition \
#                   --curr_fold=$curr \
#                   --task=3 \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-5 \
#                   --dropout=0.3 \
#                   --seed=3 \
#                   --epochs=10 \
#                   --batch_size=16 \
#                   --acc_grad=2 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.9 \
#                   --split_ratio=1 \
#                   --fair_lambda=1 \
#                   --device="cuda"
# done
# done
# done

# declare -a CANCER=("BRCA" "LUAD" "LUSC" "KIRC")
# PARTITION=(2)
# CURR=(0 1 2 3)
# SENSITIVE='{"gender": ["female", "male"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do for curr in ${CURR[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_gender/" \
#                   --partition=$partition \
#                   --curr_fold=$curr \
#                   --fair_attr="$SENSITIVE" \
#                   --task=3 \
#                   --lr=1e-5 \
#                   --dropout=0.3 \
#                   --seed=3 \
#                   --epochs=5 \
#                   --batch_size=16 \
#                   --acc_grad=2 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.9 \
#                   --split_ratio=1 \
#                   --fair_lambda=1 \
#                   --reweight \
#                   --selection="c_diff" \
#                   --device="cuda"
# done
# done
# done