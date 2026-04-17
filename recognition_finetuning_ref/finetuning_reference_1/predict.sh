#!/bin/bash

arr=(30 35 38)
doc="manuscript_one"

for i in "${arr[@]}"; do
model_name=$(printf "${doc}_%02d" $i)
test_data=$(printf "test_${doc}_%02d" $i)

python ./deep-text-recognition-benchmark/demo.py \
--Transformation None --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--workers 0 --batch_max_length 250 \
--hidden_size 512 --output_channel 512 \
--imgH 50 --imgW 2000 --PAD \
--character "0123456789!\"#$%&'()*+,-./:;<=>?@[\]^_\`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.ँंःअअंअःआइईउऊऋएऐऑओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळवशषसह़ािीुूृॅेैॉोौ्ॐ॒क़ख़ग़ज़ड़ढ़फ़ॠ।०१२३४५६७८९॰" \
--image_folder /recognition/line_images/"$test_data" --saved_model ./saved_models/v09042024/"$model_name"/best_norm_ED.pth
awk '{print $2}' log_demo_result.txt  > /recognition/line_images/"$test_data"/pred.txt
mv log_demo_result.txt tmp.txt
done