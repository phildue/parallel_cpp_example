#!/bin/bash

docker build . -t parallel_cpp_example
docker run --runtime=nvidia --rm parallel_cpp_example
