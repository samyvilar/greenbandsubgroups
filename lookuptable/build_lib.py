__author__ = 'Samy Vilar'
__date__ = 'Mar 20, 2012'

import subprocess


if __name__ == '__main__':
    subprocess.call('/usr/bin/gcc -Wall -fPIC -c *.c && /usr/bin/gcc -shared -Wl,-soname,liblut.so -o liblut.so  *.o')

