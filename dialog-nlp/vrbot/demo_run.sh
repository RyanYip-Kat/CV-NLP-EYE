#CUDA_LAUNCH_BLOCKING=1 python main.py --task meddg --super_rate 0.5 --train_stage natural  --model vrbot --train_batch_size 8 --test_batch_size 16 --device 1 --worker_num 0 --debug
##  run pass
CUDA_LAUNCH_BLOCKING=1 python main.py --task meddialog --super_rate 0.0 --model vrbot --train_batch_size 32 --test_batch_size 16 --device 1 --worker_num 0 --debug
#CUDA_LAUNCH_BLOCKING=1 python main.py --task kamed --super_rate 0.0 --model vrbot --train_batch_size 8 --test_batch_size 16 --device 2 --worker_num 0 --debug
#python main.py --task eyeai  --super_rate 0. --model vrbot --train_batch_size 32 --test_batch_size 16 --device 0 --worker_num 0 --debug --max_epoch 5 # supervised or unsupervised both passed
#test #python main.py --task meddg --test --model vrbot --ckpt data/ckpt/vrbot/fb7574e5/50-96059-train.model.ckpt --test_batch_size 32 --device 0  --worker_num 1
#python eval_main.py --eval_filename meddg_test_predictions.txt --vocab_filename data/vocabs/meddg_vocab.csv -af data/alias2sci/meddg-alias.json  > eval_meddg_metric.log
#python predict.py --task meddg -i data/eyeai_test.zip --supervised --test_batch_size 16 --device 0 --ckpt data/ckpt/VRBot/meddg/41-80000-\(0.1769-0.1039-0.0719-0.0554\).model.ckpt
#python eval_main.py -ef data/test/VRBOT-ec1b69b1-test.txt -vf data/vocabs/meddg_vocab.csv -af  data/alias2sci/meddg-alias.json
