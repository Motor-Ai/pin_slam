#!/bin/bash

echo "Build docker"

sudo docker build -f cu122.Dockerfile -t pinslam:localbuild .

echo "docker successfully build!"
