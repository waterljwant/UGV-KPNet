## Real-time keypoints detection for autonomous recovery of the unmanned ground vehicle


If you find UGV-KPNet useful in your research, please cite:

    @article{ugvkpnet,
      author    = {Li, Jie and Zhang, Sheng and Han, Kai and Yuan, Xia and Zhao, Chunxia and Liu, Yu},
      title     = {Real-time keypoints detection for autonomous recovery of the unmanned ground vehicle},
      journal   = {IET Image Processing},
      volume    = {14},
      number    = {17},
      pages     = {4690--4700},
      year      = {2021},
      publisher = {IET}
    }


### Dataset
![dataset](data/dataset_v2.png)

### Training
    python train_ShuffleNetV2_robot.py \
    --json_path='/path/to/json/robot_keypoints_20190426_640x360.json' \
    --saved_model='robotkp_LWShuffleNetV2_v1_16_cat_w1.0_v0.1.pth' \
    --model='LWShuffleNetV2_v1_16_cat' \
    --multistage 0 \
    --gpu_ids 3 \
    --epochs=400 \
    --batch_size=24 \
    --workers=4 \
    --lr=0.5 \
    --batch_size_val=4 \
    --workers_val=1 \
    --data_augment True \
    --resample True \
    --logdir='./logs/robotkp_LWShuffleNetV2_v1_16_cat_w1.0' 2>&1 |tee train_LWShuffleNetV2_v1_16_cat_w1.0_v0.1.log



Our work is built on https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation.
