python export_resnet.py temp/inception_resnet_v2/%1/all models/%1.pb
xcopy %2\%1-labels.txt models
@echo input_image > models/%1-model.txt
@echo InceptionResnetV2/Logits/Predictions >> models/%1-model.txt