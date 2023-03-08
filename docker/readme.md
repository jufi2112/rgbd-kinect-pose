Run 2 scripts to build the final docker image, your NVIDIA driver should support CUDA 10.2 or 11.3. <br> **Note that currently, only the CUDA 10.2 version will work.**
1) ~3 min, you will be prompted to accept eula manually
    - from `docker/accept_eula` dir run `./build.sh -c [10, 11]`
      - `-c 10` will build with CUDA 10.2 and `-c 11` will build with CUDA 11.3 support
2) can take more than 1 hour
    - If you want to build the CUDA 10.2 version:
      - Download libcudnn7 libraries from [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn) to `resources` directory
        - `libcudnn7_7.6.5.32-1+cuda10.2_amd64.deb`
        - `libcudnn7-dev_7.6.5.32-1+cuda10.2_amd64.deb`
    - from `docker` dir run `./build.sh -c [10, 11]`
