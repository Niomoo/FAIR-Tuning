# Cancer classification for example
declare -a CANCER=("BRCA" "LUAD LUSC" "KIRP KIRC KICH")
PARTITION=(1)
SENSITIVE='{"race": ["white", "black or african american"]}'

for cancer in "${CANCER[@]}";
do for partition in ${PARTITION[@]};
do python main_base.py --cancer $cancer \
                  --model_path="./models/" \
                  --partition=$partition \
                  --fair_attr="$SENSITIVE" \
                  --task=1 \
                  --lr=1e-5 \
                  --dropout=0.3 \
                  --seed=0 \
                  --epochs=10 \
                  --batch_size=16 \
                  --acc_grad=2 \
                  --scheduler_step=10 \
                  --scheduler_gamma=0.9 \
                  --device="cuda"
    done
    done


declare -a CANCER=("BRCA" "LUAD LUSC" "KIRP KIRC KICH")
PARTITION=(1)
SENSITIVE='{"race": ["white", "black or african american"]}'

for cancer in "${CANCER[@]}";
do for partition in ${PARTITION[@]};
do python main_base.py --cancer $cancer \
                  --model_path="./models/" \
                  --weight_path="" \
                  --partition=$partition \
                  --fair_attr="$SENSITIVE" \
                  --task="1" \
                  --lr=1e-6 \
                  --dropout=0.3 \
                  --seed=0 \
                  --epochs=10 \
                  --batch_size=16 \
                  --acc_grad=2 \
                  --scheduler_step=10 \
                  --scheduler_gamma=0.9 \
                  --fair_lambda=1 \
                  --constraint="" \
                  --reweight \
                  --selection="avgEOpp" \
                  --device="cuda"
    done
    done

declare -a CANCER=("BRCA" "LUAD LUSC" "KIRP KIRC KICH")
PARTITION=(2)
CURR=(0 1 2 3)
SENSITIVE='{"race": ["white", "black or african american"]}'

for cancer in "${CANCER[@]}";
do for partition in ${PARTITION[@]};
do for curr in ${CURR[@]};
do python main_base.py --cancer $cancer \
                  --model_path="./models/" \
                  --partition=$partition \
                  --curr_fold=$curr \
                  --fair_attr="$SENSITIVE" \
                  --task=1 \
                  --lr=1e-5 \
                  --dropout=0.3 \
                  --seed=0 \
                  --epochs=5 \
                  --batch_size=16 \
                  --acc_grad=2 \
                  --scheduler_step=10 \
                  --scheduler_gamma=0.9 \
                  --device="cuda"
done
done
done

declare -a CANCER=("BRCA" "LUAD LUSC" "KIRP KIRC KICH")
PARTITION=(2)
CURR=(0 1 2 3)
SENSITIVE='{"race": ["white", "black or african american"]}'

for cancer in "${CANCER[@]}";
do for partition in ${PARTITION[@]};
do for curr in ${CURR[@]};
do python main_base.py --cancer $cancer \
                  --model_path="./models/" \
                  --weight_path="" \
                  --partition=$partition \
                  --curr_fold=$curr \
                  --fair_attr="$SENSITIVE" \
                  --task="1" \
                  --lr=1e-6 \
                  --dropout=0.3 \
                  --seed=0 \
                  --epochs=5 \
                  --batch_size=16 \
                  --acc_grad=2 \
                  --scheduler_step=10 \
                  --scheduler_gamma=0.9 \
                  --fair_lambda=1 \
                  --constraint="" \
                  --reweight \
                  --selection="avgEOpp" \
                  --device="cuda"
done
done
done