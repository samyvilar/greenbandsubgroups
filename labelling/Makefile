

build:
	gcc -Wall -fPIC -O2 -c labelling.c
	gcc -shared -Wl -o liblabelling.so labelling.o

clean:
	rm *.o
	rm *.so
