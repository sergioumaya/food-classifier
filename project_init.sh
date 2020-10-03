#!/bin/bash
sudo docker run --gpus all -it -v /home/sergio/Documents/FoodClassifier:/tf/FoodClassifier -p 8888:8888 food_classifer_project:latest