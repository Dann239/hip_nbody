hipify-clang -o=kernel.cpp ../kernel.cu &&\
hipcc kernel.cpp &&\
./a.out
rm a.out