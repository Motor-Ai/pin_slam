#!/bin/bash

INPUT_DIR="/home/thomas/Downloads/uber_arena/frames"
LIDAR_CALIB="/home/thomas/Downloads/exp_data_collection_test/calibration"
OUTPUT_DIR="/home/thomas/Documents/coding/lidar_cam_metric/PIN_SLAM/runs_docker"
CONFIG=./config/lidar_slam/run_ivu2_loader_time_color.yaml

sudo docker run --rm -it --gpus all \
  -v $INPUT_DIR:/input_data \
  -v $OUTPUT_DIR:/output_data \
  -v $LIDAR_CALIB:/calib \
  pinslam:localbuild \
  bash -c "cd /pin_slam && python pin_slam.py $CONFIG -i /input_data -o /output_data --lidar-calib-path /calib/lidars.yaml -smd"