export CC = gcc
export CXX = g++
export CFLAGS = -g

export BINPATH = ../bin/

OBJS = matrix.o gbdt.o tasker.o
BIN = main

.PHONY: clean

$(BINPATH)$(BIN): $(OBJS) test_main.cpp
	$(CXX) $(CFLAGS) -o $@ $^

tasker.o: tasker.cpp gbdt.h matrix.h
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp, $^))

gbdt.o: gbdt.cpp gbdt.h basic_func.h Config.h matrix.h tree.h  types.h 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp, $^))

matrix.o: matrix.cpp matrix.h basic_func.h Config.h gbdt.h 
	$(CXX) -c $(CFLAGS) -o $@ $(filter %.cpp, $^)


clean:
	-rm -f *.o
	-rm -f main
