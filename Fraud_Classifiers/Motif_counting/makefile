tmc: main.o tmc.o
	clang++ -std=c++11 main.o tmc.o -o TMC

main.o: main.cpp
	clang++ -c -std=c++11 main.cpp

tmc.o: tmc.cpp
	clang++ -c -std=c++11 tmc.cpp

clean:
	rm *.o TMC