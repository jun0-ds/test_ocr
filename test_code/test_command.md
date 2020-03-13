#### text detection
python test.py --trained_model=./weights/craft_mlt_25k.pth --test_folder=../data/test_folder --refiner_model=./weights/craft_refiner_CTW1500.pth


#### text interprete
정확도 높은 모델
CUDA_VISIBLE_DEVICES=0 python demo_edit.py \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--image_folder ../data/test_image/ \
--sensitive \
--saved_model ./weights/TPS-ResNet-BiLSTM-Attn-Seed190702001/best_accuracy.pth

정확도 낮은 모델
CUDA_VISIBLE_DEVICES=0 python demo_edit.py \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--image_folder ../data/test_image/ \
--sensitive \
--saved_model ./weights/TPS-ResNet-BiLSTM-Attn-Seed190701001/best_accuracy.pth
