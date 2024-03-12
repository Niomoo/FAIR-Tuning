declare -a CANCER=("BRCA" "LUAD LUSC" "KIRP KIRC KICH")
PARTITION=(1)
SENSITIVE='{"race": ["white", "black or african american"]}'

for cancer in "${CANCER[@]}";
do for partition in ${PARTITION[@]};
do python inference.py --cancer $cancer \
                  --model_path="./models/" \
                  --partition=$partition \
                  --fair_attr="$SENSITIVE" \
                  --task=1 \
                  --seed=0 \
                  --device="cuda"
    done
    done

declare -a CANCER=("BRCA" "LUAD LUSC" "KIRP KIRC KICH")
PARTITION=(1)
SENSITIVE='{"race": ["white", "black or african american"]}'

for cancer in "${CANCER[@]}";
do for partition in ${PARTITION[@]};
do python inference.py --cancer $cancer \
                  --model_path="./models/" \
                  --partition=$partition \
                  --fair_attr="$SENSITIVE" \
                  --task=1 \
                  --seed=0 \
                  --reweight \
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
do python inference.py --cancer $cancer \
                  --model_path="./models/" \
                  --weight_path=$curr \
                  --partition=$partition \
                  --fair_attr="$SENSITIVE" \
                  --task=1 \
                  --seed=0 \
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
do python inference.py --cancer $cancer \
                  --model_path="./models/" \
                  --weight_path=$curr \
                  --partition=$partition \
                  --fair_attr="$SENSITIVE" \
                  --task=1 \
                  --seed=0 \
                  --reweight \
                  --device="cuda"
    done
    done
    done