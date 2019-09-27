mkdir hipify
cd hipify

HIP_R=$(pwd)

mkdir git build dist

export MAKEFLAGS="-j$(nproc)"

function make_install {
    git clone --depth 1 https://github.com/$2/$1.git $HIP_R/git/$1 --recursive
    mkdir $1
    cd $1
    cmake -DCMAKE_PREFIX_PATH=$HIP_R/dist -DCMAKE_INSTALL_PREFIX=$HIP_R/dist -DCMAKE_BUILD_TYPE=Release $HIP_R/git/$1
    cmake --build . --target install
    cd ..
}

make_install ROCT-Thunk-Interface RadeonOpenCompute

git clone --depth 1 https://github.com/RadeonOpenCompute/ROCR-Runtime.git $HIP_R/git/ROCR-Runtime --recursive
cp -r $HIP_R/git/ROCT-Thunk-Interface/include/* -t $HIP_R/git/ROCR-Runtime/src/core/inc
mkdir ROCR-Runtime
cd ROCR-Runtime
cmake -DCMAKE_PREFIX_PATH=$HIP_R/dist -DCMAKE_INSTALL_PREFIX=$HIP_R/dist -DCMAKE_BUILD_TYPE=Release $HIP_R/git/ROCR-Runtime/src
cmake --build . --target install
cd ..

make_install llvm ROCm-Developer-Tools
make_install clang ROCm-Developer-Tools

make_install hcc RadeonOpenCompute
make_install HIP ROCm-Developer-Tools

mkdir hipify
cd hipify
cmake -DCMAKE_PREFIX_PATH=$HIP_R/dist -DCMAKE_INSTALL_PREFIX=$HIP_R/dist -DCMAKE_BUILD_TYPE=Release $HIP_R/git/HIP/hipify-clang
cmake --build . --target install

