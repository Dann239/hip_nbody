cp ../main.cpp main.cpp
cp ../kernel.h kernel.h
rm a.out
hipify-clang -o=kernel.cpp ../kernel.cu &&\
hipcc main.cpp kernel.cpp &&\
./a.out
