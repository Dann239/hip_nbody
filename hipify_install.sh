mkdir hipify
cd hipify
mkdir git cmake
cd git

git clone --depth 1 https://github.com/ROCm-Developer-Tools/HIP.git
git clone --depth 1 https://github.com/llvm-mirror/llvm.git
git clone --depth 1 https://github.com/llvm-mirror/clang.git

cd ../cmake
mkdir build_llvm build_clang build_hipify dist_llvm_clang dist_hipify

export MAKEFLAGS="-j$(nproc)"

cd build_llvm
cmake -DCMAKE_INSTALL_PREFIX=../dist_llvm_clang -DCMAKE_BUILD_TYPE=Release ../../git/llvm
cmake --build . --target install

cd ../build_clang
cmake -DCMAKE_PREFIX_PATH=$(pwd)/../dist_llvm_clang -DCMAKE_INSTALL_PREFIX=../dist_llvm_clang -DCMAKE_BUILD_TYPE=Release ../../git/clang
cmake --build . --target install

cd ../build_hipify
cmake -DCMAKE_PREFIX_PATH=$(pwd)/../dist_llvm_clang -DCMAKE_INSTALL_PREFIX=../dist_hipify -DCMAKE_BUILD_TYPE=Release ../../git/HIP/hipify-clang
cmake --build . --target install
