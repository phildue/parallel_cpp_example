#!/bin/bash
git clone --branch 3.4 https://gitlab.com/libeigen/eigen.git
cd eigen
# https://forums.developer.nvidia.com/t/possible-bug-in-nvc-23-05/260553
sed -i 's/#if EIGEN_COMP_CLANG || EIGEN_COMP_CASTXML/#if EIGEN_COMP_CLANG || EIGEN_COMP_CASTXML || __NVCOMPILER_LLVM__/' ./Eigen/src/Core/arch/NEON/Complex.h
mkdir build
cd build
cmake ..
make -j2
make install
cd ..
rm build -r
