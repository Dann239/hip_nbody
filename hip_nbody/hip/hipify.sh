mkdir /tmp/nbody 2>/dev/null
cp ../*.c ../*.h ../*.cpp ../*.hpp ../*.cuh -t /tmp/nbody 2>/dev/null
rm a.out 2>/dev/null
hipify-perl ../kernel.cu > /tmp/nbody/kernel.cpp &&\
hipcc /tmp/nbody/*.cpp -std=c++14 -o /tmp/nbody/nbody.run &&\
/tmp/nbody/nbody.run
