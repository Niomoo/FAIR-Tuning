# declare -a CANCER=("BRCA")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_split/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-5 \
#                   --dropout=0.2 \
#                   --seed=0 \
#                   --epochs=200 \
#                   --batch_size=16 \
#                   --acc_grad=2 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.7 \
#                   --split_ratio=1 \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("BRCA")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_split/" \
#                   --weight_path="6" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-6 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=300 \
#                   --batch_size=16 \
#                   --acc_grad=1 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.8 \
#                   --split_ratio=1 \
#                   --fair_lambda=1 \
#                   --constraint="MMF" \
#                   --reweight \
#                   --rank=4 \
#                   --selection="avgEOpp" \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("BRCA")
# PARTITION=(2)
# CURR=(0 1 2 3)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do for curr in ${CURR[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_split/" \
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
#                   --scheduler_gamma=0.9 \
#                   --split_ratio=1 \
#                   --device="cuda"
# done
# done
# done

# declare -a CANCER=("BRCA")
# PARTITION=(2)
# CURR=(0 1 2 3)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do for curr in ${CURR[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_split/" \
#                   --weight_path="4" \
#                   --partition=$partition \
#                   --curr_fold=$curr \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-6 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=300 \
#                   --batch_size=16 \
#                   --acc_grad=1 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.8 \
#                   --split_ratio=1 \
#                   --fair_lambda=1 \
#                   --constraint="AE" \
#                   --reweight \
#                   --rank=4 \
#                   --selection="EOdd" \
#                   --device="cuda"
# done
# done
# done

# ###################### LUAD/LUSC ########################################

# declare -a CANCER=("LUAD LUSC")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_split/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-6 \
#                   --dropout=0.2 \
#                   --seed=0 \
#                   --epochs=200 \
#                   --batch_size=16 \
#                   --acc_grad=2 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.8 \
#                   --split_ratio=1 \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("LUAD LUSC")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_split/" \
#                   --weight_path="6" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=5e-6 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=300 \
#                   --batch_size=16 \
#                   --acc_grad=1 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.9 \
#                   --split_ratio=1 \
#                   --fair_lambda=1 \
#                   --constraint="MMF" \
#                   --reweight \
#                   --rank=4 \
#                   --selection="avgEOpp" \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("LUAD LUSC")
# PARTITION=(2)
# CURR=(0 1 2 3)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do for curr in ${CURR[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_split/" \
#                   --partition=$partition \
#                   --curr_fold=$curr \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-6 \
#                   --dropout=0.2 \
#                   --seed=0 \
#                   --epochs=200 \
#                   --batch_size=16 \
#                   --acc_grad=1 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.9 \
#                   --split_ratio=1 \
#                   --device="cuda"
# done
# done
# done

# declare -a CANCER=("LUAD LUSC")
# PARTITION=(2)
# CURR=(0 1 2 3)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do for curr in ${CURR[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_split/" \
#                   --partition=$partition \
#                   --curr_fold=$curr \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=5e-6 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=300 \
#                   --batch_size=16 \
#                   --acc_grad=1 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.9 \
#                   --split_ratio=1 \
#                   --fair_lambda=1 \
#                   --constraint="AE" \
#                   --reweight \
#                   --rank=4 \
#                   --selection="avgEOpp" \
#                   --device="cuda"
# done
# done
# done

# ###################### KIRP/KIRC/KICH ########################################

# declare -a CANCER=("KIRP KIRC KICH")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_split/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-6 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=300 \
#                   --batch_size=16 \
#                   --acc_grad=3 \
#                   --scheduler_step=20 \
#                   --scheduler_gamma=0.8 \
#                   --split_ratio=1 \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("KIRP KIRC KICH")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_split/" \
#                   --weight_path="6" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-6 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=300 \
#                   --batch_size=16 \
#                   --acc_grad=1 \
#                   --scheduler_step=5 \
#                   --scheduler_gamma=0.8 \
#                   --split_ratio=1 \
#                   --fair_lambda=1 \
#                   --constraint="MMF" \
#                   --reweight \
#                   --rank=4 \
#                   --selection="avgEOpp" \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("KIRP KIRC KICH")
# PARTITION=(2)
# CURR=(0 1 2 3)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do for curr in ${CURR[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_split/" \
#                   --partition=$partition \
#                   --curr_fold=$curr \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-6 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=300 \
#                   --batch_size=16 \
#                   --acc_grad=1 \
#                   --scheduler_step=15 \
#                   --scheduler_gamma=0.8 \
#                   --split_ratio=1 \
#                   --device="cuda"
# done
# done
# done

# declare -a CANCER=("KIRP KIRC KICH")
# PARTITION=(2)
# CURR=(0 1 2 3)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do for curr in ${CURR[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_split/" \
#                   --partition=$partition \
#                   --curr_fold=$curr \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-6 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=300 \
#                   --batch_size=16 \
#                   --acc_grad=1 \
#                   --scheduler_step=5 \
#                   --scheduler_gamma=0.8 \
#                   --split_ratio=1 \
#                   --fair_lambda=1 \
#                   --constraint="AE" \
#                   --reweight \
#                   --rank=4 \
#                   --selection="avgEOpp" \
#                   --device="cuda"
# done
# done
# done

###################### COAD/READ ########################################
# declare -a CANCER=("COAD READ")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_split/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-6 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=200 \
#                   --batch_size=16 \
#                   --acc_grad=2 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.9 \
#                   --split_ratio=1 \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("COAD READ")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_split/" \
#                   --weight_path="" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-6 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=300 \
#                   --batch_size=16 \
#                   --acc_grad=1 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=0.9 \
#                   --split_ratio=1 \
#                   --fair_lambda=1 \
#                   --constraint="MMF" \
#                   --reweight \
#                   --rank=4 \
#                   --selection="EOdd" \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("COAD READ")
# PARTITION=(2)
# CURR=(0 1 2 3)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do for curr in ${CURR[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_split/" \
#                   --partition=$partition \
#                   --curr_fold=$curr \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-6 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=300 \
#                   --batch_size=16 \
#                   --acc_grad=1 \
#                   --scheduler_step=10 \
#                   --scheduler_gamma=1 \
#                   --split_ratio=1 \
#                   --device="cuda"
# done
# done
# done

# declare -a CANCER=("COAD READ")
# PARTITION=(2)
# CURR=(0 1 2 3)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do for curr in ${CURR[@]};
# do python main_base.py --cancer $cancer \
#                   --model_path="./models_split/" \
#                   --partition=$partition \
#                   --curr_fold=$curr \
#                   --fair_attr="$SENSITIVE" \
#                   --lr=1e-6 \
#                   --dropout=0.3 \
#                   --seed=0 \
#                   --epochs=300 \
#                   --batch_size=16 \
#                   --acc_grad=1 \
#                   --scheduler_step=5 \
#                   --scheduler_gamma=0.8 \
#                   --split_ratio=1 \
#                   --fair_lambda=1 \
#                   --constraint="" \
#                   --reweight \
#                   --rank=4 \
#                   --selection="avgEOpp" \
#                   --device="cuda"
# done
# done
# done


# declare -a CANCER=("COAD READ")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference.py --cancer $cancer \
#                   --model_path="./models_split/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --seed=0 \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("COAD READ")
# PARTITION=(1)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference.py --cancer $cancer \
#                   --model_path="./models_split/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --reweight \
#                   --seed=0 \
#                   --device="cuda"
#     done
#     done

# declare -a CANCER=("COAD READ")
# PARTITION=(2)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference.py --cancer $cancer \
#                   --model_path="./models_split/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --seed=0 \
#                   --device="cuda"
#     done
#     done


# declare -a CANCER=("BRCA" "LUAD LUSC" "KIRP KIRC KICH")
# PARTITION=(2)
# SENSITIVE='{"race": ["white", "black or african american"]}'

# for cancer in "${CANCER[@]}";
# do for partition in ${PARTITION[@]};
# do python inference.py --cancer $cancer \
#                   --model_path="./models_split/" \
#                   --partition=$partition \
#                   --fair_attr="$SENSITIVE" \
#                   --seed=0 \
#                   --reweight \
#                   --device="cuda"
#     done
#     done
