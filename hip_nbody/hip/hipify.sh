cp ../main.cpp main.cpp
cp ../kernel.h kernel.h
cp ../window.h window.h
cp ../properties.h properties.h
rm a.out
hipify-clang --extra-arg="-std=c++14" -o=kernel.cpp ../kernel.cu &&\
hipcc main.cpp kernel.cpp -std=c++14 &&\
./a.out
