cp ../main.cpp main.cpp
cp ../kernel.h kernel.h
rm a.out
hipify-clang --extra-arg="-std=c++14" ../kernel.cu
hipcc main.cpp kernel.cpp -std=c++14 &&\
./a.out
