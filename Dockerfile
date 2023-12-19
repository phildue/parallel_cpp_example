FROM nvcr.io/nvidia/nvhpc:23.11-devel-cuda_multi-ubuntu22.04
ADD . /workspace
RUN mkdir -p /workspace/build && cd /workspace/build && cmake .. && make
ADD entrypoint.sh /entrypoint.sh
ENTRYPOINT /entrypoint.sh
