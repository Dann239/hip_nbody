mkdir hipify
cd hipify

HIP_R=$(pwd)

mkdir git cmake build
cd cmake

export MAKEFLAGS="-j$(nproc)"

function make_install {
    mkdir $1
    cd $1
    cd ..
}

mkdir llvm
cd llvm
git clone --depth 1 https://github.com/ROCm-Developer-Tools/llvm.git $HIP_R/git/llvm --recursive

cmake \
\
-D CMAKE_PREFIX_PATH=$HIP_R/build \
-D CMAKE_INSTALL_PREFIX=$HIP_R/build \
-D CMAKE_BUILD_TYPE=Release \
-D LLVM_TARGETS_TO_BUILD="X86;NVPTX" \
-D LLVM_ENABLE_PROJECTS="" \
-D LLVM_BUILD_TOOLS=OFF \
-D LLVM_INCLUDE_TOOLS=OFF \
-D LLVM_INCLUDE_EXAMPLES=OFF \
-D LLVM_INCLUDE_TESTS=OFF \
-D LLVM_INCLUDE_BENCHMARKS=OFF \
\
$HIP_R/git/llvm

cmake --build . --target install
cd ..

mkdir clang
cd clang
git clone --depth 1 https://github.com/ROCm-Developer-Tools/clang.git $HIP_R/git/clang --recursive
cmake -DCMAKE_PREFIX_PATH=$HIP_R/build -DCMAKE_INSTALL_PREFIX=$HIP_R/build -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" $HIP_R/git/clang
cmake --build . --target install
cd ..

mkdir HIP
cd HIP
git clone --depth 1 https://github.com/ROCm-Developer-Tools/HIP.git $HIP_R/git/HIP --recursive
cmake -DCMAKE_PREFIX_PATH=$HIP_R/build -DCMAKE_INSTALL_PREFIX=$HIP_R/build -DCMAKE_BUILD_TYPE=Release $HIP_R/git/HIP
cmake --build . --target install
cd ..

mkdir hipify
cd hipify
cmake -DCMAKE_PREFIX_PATH=$HIP_R/build -DCMAKE_INSTALL_PREFIX=$HIP_R/build/bin -DCMAKE_BUILD_TYPE=Release $HIP_R/git/HIP/hipify-clang
cmake --build . --target install
cd ..

