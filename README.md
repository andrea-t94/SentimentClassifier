- Using native python 3.9.14 on linux ARM64/v8 since it is native ARM64 platform, but no MacOS image support
- Using PyTorch stable version because there is no support on Docker for MPS --> thus only useful for inference
- Used old version of PyTorch and torchtext/torchdata (1.11.0 and compatibles) because of lack of support for 
PyTorch versions for LinuxARM64 (aarch84) --> see [here](https://anaconda.org/pytorch/)

Code used
docker buildx build --platform linux/arm64 --tag test_docker_on_m1 .

see [here](https://blog.codecentric.de/python-on-m1-chip-running-smoothly-using-docker) for a way to encapsulating x64
docker images (also over a ARM64 env) to be run on every architecture. Uses Poetry