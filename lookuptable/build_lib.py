__author__ = 'Samy Vilar'
__date__ = 'Mar 20, 2012'

import subprocess


if __name__ == '__main__':
    subprocess.call('gcc -Wall -fPIC -c *.c && gcc -shared -Wl,-soname,liblut.so -o liblut.so  *.o')

