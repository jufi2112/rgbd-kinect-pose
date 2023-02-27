Run 2 scripts to build final docker image, your nvidia driver should support cuda 10.2 or 11.3:
1) ~3 min, you will be prompted to accept eula manually
    - If you want to build the CUDA 10.2 version:
      - Updating the CUDA Linux GPG repository key: Download keyring to `docker/accept_eula`, e.g. `wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb`
    - from `docker/accept_eula` dir run `./build.sh -c [10, 11]`
      - `-c 10` will build for CUDA 10.2 and `-c 11` will build for CUDA 11.3
2) can take more than 1 hour
    If you want to build the CUDA 10.2 version:
      - Download libcudnn7 libraries from [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn) to `resources` directory
        - `libcudnn7_7.6.5.32-1+cuda10.2_amd64.deb`
        - `libcudnn7-dev_7.6.5.32-1+cuda10.2_amd64.deb`
    - from `docker` dir run `./build.sh -c [10, 11]`
