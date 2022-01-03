# shellcheck disable=SC1009
python main.py \
  --model_type phobert \
  --model_name_or_path  vinai/phobert-base \
  --do_train \
  --do_lower_case \
  --model_dir checkpoint \
  --train_file train_data_model.json \
  --train_batch_size 12 \
  --max_question_len 50 \
  --max_seq_len 256 \
  --learning_rate 3e-5 \
  --num_train_epochs 6

