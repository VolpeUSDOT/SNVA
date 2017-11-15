mkdir models\MobilenetV1
python export_mobilenet.py temp/MobilenetV1/%1/all models/MobilenetV1/%1.pb
xcopy %2\%1-labels.txt models\MobilenetV1
@echo input_image > models/MobilenetV1/%1-model.txt
@echo MobilenetV1/Predictions/Reshape_1 >> models/MobilenetV1/%1-model.txt