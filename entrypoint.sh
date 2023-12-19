#!/bin/bash

echo "Running executable without -stdpar (CPU)"
/workspace/build/parallel_cpp_example

echo "Running executable with -stdpar (GPU)"
/workspace/build/parallel_cpp_example_stdpar
