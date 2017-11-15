mkdir models\InceptionResnetV2
python export_resnet.py temp/inception_resnet_v2/%1/all models/InceptionResnetV2/%1.pb
xcopy %2\%1-labels.txt models\InceptionResnetV2
@echo input_image > models/InceptionResnetV2/%1-model.txt
@echo InceptionResnetV2/Logits/Predictions >> models/InceptionResnetV2/%1-model.txt