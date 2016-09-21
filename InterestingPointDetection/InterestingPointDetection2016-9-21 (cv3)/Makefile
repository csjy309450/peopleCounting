LIBS_OPENCV = `$(PREFIX)pkg-config --libs opencv`
INCLUDE_OPENCV = `$(PREFIX)pkg-config --cflags opencv`

myMain: myMain.cpp MatControler.o InterestingPointDetector.o
	g++ -O3 -Wall -Werror -pedantic $(INCLUDE_OPENCV) myMain.cpp MatControler.o \
		InterestingPointDetector.o -ggdb $(LIBS_OPENCV) -o myMain

InterestingPointDetector.o: InterestingPointDetector.cpp MatControler.o
	g++ -O3 -Wall -Werror -pedantic $(INCLUDE_OPENCV) InterestingPointDetector.cpp MatControler.o \
		-c -ggdb $(LIBS_OPENCV) -o InterestingPointDetector.o

MatControler.o:MatControler.cpp
	g++ -O3 -Wall -Werror -pedantic $(INCLUDE_OPENCV) MatControler.cpp -c -ggdb $(LIBS_OPENCV) -o \
		MatControler.o
