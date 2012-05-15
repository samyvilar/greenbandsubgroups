

build:
	gcc -Wall -fPIC -c -ggdb labelling.c
	gcc -shared -Wl -o liblabelling.so labelling.o

clean:
	rm *.o
	rm *.so
