mkdir /tmp/nbody 2>/dev/null
cp ../*.c ../*.h ../*.cpp ../*.hpp ../*.cuh -t /tmp/nbody 2>/dev/null
hipify-clang --extra-arg="-std=c++14" -o=/tmp/nbody/kernel.cpp ../kernel.cu &&\
hipcc /tmp/nbody/*.cpp -std=c++14 -o=/tmp/nbody/nbody.run --run
