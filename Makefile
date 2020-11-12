CC = g++-4.8

TARGETS=multilayer

all: $(TARGETS)

neural: main.cpp
	 g++ -g -Ofast -o neural main.cpp -fopenmp -lblas -lcblas

multilayer: multilayer.cpp
	 g++ -g -Ofast -o multilayer multilayer.cpp -lblas -lcblas

multilayer_parallel: multilayer_parallel.cpp
	 g++ -g -Ofast -o multilayer_parallel multilayer_parallel.cpp -fopenmp -lblas -lcblas

clean:
	rm -f *.o $(TARGETS)