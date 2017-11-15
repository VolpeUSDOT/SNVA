del /Q temp\MobilenetV1\%1\*.*
del /Q temp\MobilenetV1\%1\all\*.*
python download_and_convert_data.py --dataset_name=%1 --dataset_dir=%2
python train_image_classifier.py --train_dir=temp/MobilenetV1/%1 --dataset_name=%1 --dataset_split_name=train --dataset_dir=%2 --model_name=mobilenet_v1 --checkpoint_path=checkpoints/mobilenet_v1_1.0_224.ckpt --checkpoint_exclude_scopes=MobilenetV1/Logits,MobilenetV1/AuxLogits --trainable_scopes=MobilenetV1/Logits,MobilenetV1/AuxLogits --max_number_of_steps=3500 --batch_size=128 --learning_rate=0.01 --learning_rate_decay_type=fixed --save_interval_secs=60 --save_summaries_secs=30 --log_every_n_steps=50 --optimizer=sgd --weight_decay=0.00004
python train_image_classifier.py --train_dir=temp/MobilenetV1/%1/all --dataset_name=%1 --dataset_split_name=train --dataset_dir=%2 --model_name=mobilenet_v1 --checkpoint_path=temp/MobilenetV1/%1 --max_number_of_steps=1000 --batch_size=32 --learning_rate=0.0001 --learning_rate_decay_type=fixed --save_interval_secs=60 --save_summaries_secs=60 --log_every_n_steps=10 --optimizer=rmsprop --weight_decay=0.00004
python eval_image_classifier.py --checkpoint_path=temp/MobilenetV1/%1/all --eval_dir=temp/MobilenetV1/%1/all --dataset_name=%1 --dataset_split_name=validation --dataset_dir=%2 --model_name=mobilenet_v1