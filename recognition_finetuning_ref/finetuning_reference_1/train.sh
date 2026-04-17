#!/bin/bash
# to be run from the root directory

doc="manuscript_name"
arr=(30 35 38)

# Create LMDB dataset
python3 deep-text-recognition-benchmark/create_lmdb_dataset.py --inputPath ./recognition/line_images/val_$doc --gtFile ./recognition/line_images/val_$doc/labels.txt --outputPath ./recognition/line_images/val_$doc/lmbd_output_val

# Train the model
for i in "${arr[@]}"; do
    model_name=$(printf "${doc}_%02d" $i)
    train_data=$(printf "train_${doc}_%02d" $i)

    python deep-text-recognition-benchmark/create_lmdb_dataset.py --inputPath ./recognition/line_images/$train_data --gtFile ./recognition/line_images/$train_data/labels.txt --outputPath ./recognition/line_images/$train_data/lmbd_output_train --checkValid False

    python deep-text-recognition-benchmark/train.py \
    --train_data recognition/line_images/$train_data --valid_data recognition/line_images/val_$doc --data_filtering_off \
    --select_data recognition/line_images/$train_data --batch_ratio 1 --batch_size 16 \
    --workers 0 --batch_max_length 250 --valInterval 5 --num_iter 2000 \
    --hidden_size 512 --output_channel 512 \
    --imgH 50 --imgW 1150 --PAD \
    --exp_name recognition/$model_name \
    --character "0123456789!\"#$%&'()*+,-./:;<=>?@[\]^_\`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.ँंःअअंअःआइईउऊऋएऐऑओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळवशषसह़ािीुूृॅेैॉोौ्ॐ॒क़ख़ग़ज़ड़ढ़फ़ॠ।०१२३४५६७८९॰" \
    --Transformation None --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
    --saved_model deep-text-recognition-benchmark/devanagari.pth
done
