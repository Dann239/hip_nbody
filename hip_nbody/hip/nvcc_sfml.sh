nvcc ../*.cpp ../*.cu -o /tmp/nvcc_nbody.run --run --define-macro SFML_STATIC -Xlinker -lsfml-graphics,-lsfml-window,-lsfml-system
