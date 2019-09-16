mkdir hipify
cd hipify

HIP_R=$(pwd)

mkdir git cmake
cd git

git clone --depth 1 https://github.com/ROCm-Developer-Tools/HIP.git
git clone --depth 1 https://github.com/ROCm-Developer-Tools/llvm.git
git clone --depth 1 https://github.com/ROCm-Developer-Tools/clang.git

cd ../cmake

mkdir build dist
mkdir build/llvm build/clang build/hip build/hipify
mkdir dist/llvm dist/clang dist/hip

cd build
export MAKEFLAGS="-j$(nproc)"

cd llvm
cmake -DCMAKE_INSTALL_PREFIX=$HIP_R/cmake/dist/llvm -DCMAKE_BUILD_TYPE=Release $HIP_R/git/llvm
cmake --build . --target install
cd ..

cd clang
cmake -DCMAKE_PREFIX_PATH=$HIP_R/cmake/dist/llvm -DCMAKE_INSTALL_PREFIX=$HIP_R/cmake/dist/llvm -DCMAKE_BUILD_TYPE=Release $HIP_R/git/clang
cmake --build . --target install
cd ..

cd hip
cmake -DCMAKE_PREFIX_PATH="$HIP_R/cmake/dist/llvm;$HIP_R/cmake/dist/clang" -DCMAKE_INSTALL_PREFIX=$HIP_R/cmake/dist/hip -DCMAKE_BUILD_TYPE=Release $HIP_R/git/HIP
cmake --build . --target install
cd ..

cd hipify
cmake -DCMAKE_PREFIX_PATH="$HIP_R/cmake/dist/llvm;$HIP_R/cmake/dist/clang" -DCMAKE_INSTALL_PREFIX=$HIP_R/cmake/dist/hip -DCMAKE_BUILD_TYPE=Release $HIP_R/git/HIP/hipify-clang
cmake --build . --target install
cd ..
