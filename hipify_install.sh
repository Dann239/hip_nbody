mkdir hipify
cd hipify

HIP_R=$(pwd)

mkdir git cmake
cd cmake

export MAKEFLAGS="-j$(nproc)"

function make_install {
    git clone --depth 1 https://github.com/$2/$1.git $HIP_R/git/$1 --recursive
    mkdir $1
    cd $1
    cmake -DCMAKE_PREFIX_PATH=~/.local -DCMAKE_INSTALL_PREFIX=~/.local -DCMAKE_BUILD_TYPE=Release $HIP_R/git/$1
    cmake --build . --target install
    cd ..
}

make_install llvm ROCm-Developer-Tools
make_install clang ROCm-Developer-Tools
make_install HIP ROCm-Developer-Tools

mkdir hipify
cd hipify
cmake -DCMAKE_PREFIX_PATH=~/.local -DCMAKE_INSTALL_PREFIX=~/.local -DCMAKE_BUILD_TYPE=Release $HIP_R/git/HIP/hipify-clang
cmake --build . --target install

