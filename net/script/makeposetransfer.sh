source poseenv/bin/activate

python3 test.py --dataroot ./market_data/ --name market_PATN --model PATN --phase test --dataset_mode keypoint --norm batch --batchSize 1 --resize_or_crop no --gpu_ids 2 --BP_input_nc 18 --no_flip --which_model_netG PATN --checkpoints_dir ./checkpoints --pairLst ./market_data/market-pairs-test.csv --which_epoch latest --results_dir ./results --display_id 0

deactivate