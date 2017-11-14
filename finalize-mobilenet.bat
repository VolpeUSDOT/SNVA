python export_mobilenet.py temp/MobilenetV1/%1/all models/%1.pb
xcopy %2\%1-labels.txt models
@echo input_image > models/%1-model.txt
@echo MobilenetV1/Predictions/Reshape_1 >> models/%1-model.txt