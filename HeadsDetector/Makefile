LIBS_OPENCV=`$(PREFIX)pkg-config --libs opencv`
INCLUDE_OPENCV=`$(PREFIX)pkg-config --cflags opencv`

default:
	g++ -o detectHeads $(INCLUDE_OPENCV) -O3 -ggdb detectHeads.cpp $(LIBS_OPENCV)
