mkdir hipify
cd hipify

HIP_R=$(pwd)

mkdir git build dist
cd build

export MAKEFLAGS="-j$(nproc)"

function make_install {
    git clone --depth 1 https://github.com/$2/$1.git $HIP_R/git/$1 --recursive
    mkdir $1
    cd $1
    cmake -DCMAKE_PREFIX_PATH=$HIP_R/dist -DCMAKE_INSTALL_PREFIX=$HIP_R/dist -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" $HIP_R/git/$1
    cmake --build . --target install
    cd ..
}

make_install llvm ROCm-Developer-Tools
make_install clang ROCm-Developer-Tools
make_install lld ROCm-Developer-Tools

git clone --depth 1 https://github.com/ROCm-Developer-Tools/HIP.git $HIP_R/git/HIP --recursive
mkdir HIP
cd HIP
cmake -DCMAKE_PREFIX_PATH=$HIP_R/dist -DCMAKE_INSTALL_PREFIX=$HIP_R/dist -DCMAKE_BUILD_TYPE=Release \
-DHIP_COMPILER=clang -DHIP_PATH=$HIP_R/dist -DHIP_CLANG_PATH=$HIP_R/dist/bin \
$HIP_R/git/HIP
cmake --build . --target install
cd ..

mkdir hipify
cd hipify
cmake -DCMAKE_PREFIX_PATH=$HIP_R/dist -DCMAKE_INSTALL_PREFIX=$HIP_R/dist -DCMAKE_BUILD_TYPE=Release \
-DHIP_COMPILER=clang -DHIP_PATH=$HIP_R/dist -DHIP_CLANG_PATH=$HIP_R/dist/bin $HIP_R/git/HIP/hipify-clang
cmake --build . --target install

