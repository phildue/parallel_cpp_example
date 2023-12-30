FROM nvcr.io/nvidia/nvhpc:23.11-devel-cuda_multi-ubuntu22.04 as dev

RUN apt update && apt install -y --no-install-recommends libopencv-dev

ADD install_eigen.sh /opt/
RUN cd /opt/ && . /etc/profile.d/lmod.sh && module load nvhpc-hpcx/23.11 && /opt/install_eigen.sh

FROM dev as runtime
ADD src /workspace/src
ADD CMakeLists.txt /workspace/

WORKDIR /workspace
RUN . /etc/profile.d/lmod.sh && module load nvhpc-hpcx/23.11 && mkdir -p /workspace/build && cd /workspace/build && cmake .. && make
ADD resource /workspace/resource

