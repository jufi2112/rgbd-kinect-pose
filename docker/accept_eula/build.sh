#!/usr/bin/env bash

while getopts c: flag
do
	case "${flag}" in
		c) cuda=${OPTARG};;
	esac
done

source ../source.sh
NAME_SUFFIX=""
if [[ $cuda == "10" ]]
then
	echo "Building docker image with CUDA 10.2"
	NAME_SUFFIX=$NAME_CUDA10
elif [[ $cuda == "11" ]]
then
	echo "Building docker image with CUDA 11.3"
	NAME_SUFFIX=$NAME_CUDA11
else
	echo "Specify either -c 10 for CUDA 10.2 or -c 11 for CUDA 11.3"
	exit 1
fi

NAME_01="${NAME_01}-${NAME_SUFFIX}"
NAME_02="${NAME_02}-${NAME_SUFFIX}"
docker build -t $NAME_01 -f "Dockerfile_${NAME_SUFFIX}" .
CONTAINER_ID=$(docker run -ti -d $NAME_01)
docker exec -ti $CONTAINER_ID sh -c "./install_k4abt.sh"
docker commit $CONTAINER_ID $NAME_02
docker rm -f $CONTAINER_ID
