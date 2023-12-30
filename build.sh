#!/bin/bash

docker build . -t parallel_cpp_example:dev --target dev
docker build . -t parallel_cpp_example
