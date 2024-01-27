#!/bin/bash

docker run -ti --gpus=all --rm parallel_cpp_example ./build/photometric_error "$@"
