#!/usr/bin/env bash

while getopts c: flag
do
	case "${flag}" in
		c) cuda=${OPTARG};;
	esac
done

shift
shift

source source.sh
NAME_SUFFIX=""
if [[ $cuda == "10" ]]
then
	echo "Running docker image with CUDA 10.2"
	NAME_SUFFIX=$NAME_CUDA10
elif [[ $cuda == "11" ]]
then
	echo "Running docker image with CUDA 11.3"
	NAME_SUFFIX=$NAME_CUDA11
else
	echo "Specify either -c 10 for CUDA 10.2 or -c 11 for CUDA 11.3"
	exit 1
fi

NAME_03="${NAME_03}-${NAME_SUFFIX}"

VOLUMES="-v /home/julien/git/rgbd-kinect-pose/data:/home/julien/git/rgbd-kinect-pose/data -v /home/julien/git/rgbd-kinect-pose/output:/home/julien/git/rgbd-kinect-pose/output -v $PWD/../src:/src"

# ensure nvidia is your default runtime
docker run --runtime=nvidia -ti -v /tmp/.X11-unix:/tmp/.X11-unix $PARAMS $VOLUMES $NAME_03 $@
# -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix
