#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
DATA_DIR=$1
RESULT_DIR=$2
NUM_EPOCHS=$3
LR=$4
MAX_LENGTH=$5

MODEL="xlm-mlm-xnli15-1024"
TASK='udpos'
LANGS='ar,bg,de,el,en,es,fr,hi,ru,th,tr,ur,vi,zh'

LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-xnli15-1024" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ] || [ $MODEL == "xlm-mlm-100-1280" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
fi

if [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-roberta-large" ]; then
  BATCH_SIZE=16
  GRAD_ACC=2
else
  BATCH_SIZE=32
  GRAD_ACC=1
fi

DATA_DIR=$DATA_DIR/$TASK/${TASK}_processed_maxlen${MAX_LENGTH}/
OUTPUT_DIR="$RESULT_DIR/$TASK/XLP-LR${LR}-epoch${NUM_EPOCHS}-MaxLen${MAX_LENGTH}/"
mkdir -p $OUTPUT_DIR
CUDA_VISIBLE_DEVICES=0 python third_party/ori_run_tag.py \
  --data_dir $DATA_DIR \
  --model_type $MODEL_TYPE \
  --labels $DATA_DIR/labels.txt \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUT_DIR \
  --max_seq_length  $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --gradient_accumulation_steps $GRAD_ACC \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --save_steps 500 \
  --seed 1 \
  --learning_rate $LR \
  --do_train \
  --do_predict \
  --do_predict_dev \
  --evaluate_during_training \
  --predict_langs $LANGS \
  --log_file $OUTPUT_DIR/train.log \
  --eval_all_checkpoints \
  --overwrite_output_dir \
  --save_only_best_checkpoint $LC
