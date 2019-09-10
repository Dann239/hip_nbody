cp ../main.cpp main.cpp
cp ../kernel.h kernel.h
rm a.out
hipify-clang -o=kernel.cpp ../kernel.cu -stdlib=c++14 &&\
hipcc main.cpp kernel.cpp &&\
./a.out
