

build:
	gcc -Wall -fPIC -c *.c
	gcc -shared -Wl,-soname,liblut.so -o liblut.so *.o

clean:
	rm *.o
	rm *.so

