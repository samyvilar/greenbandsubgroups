#include "lut.h"
#include <stdlib.h>
#include <stdio.h>

/* Author: Samy Vilar, create a lookup table on 4 bands, with first 3 being the indexes ...*/

void lookuptable(int *data,    /*2D DATA with n by r shape, n-1 used as indices, with DATA[n] being the sum up values.*/
                 unsigned int numrows, /* The number of rows withing the 2D data matrix. */
                 unsigned int numcols,  /* The number of columns withing the 2D data matrix */
                 unsigned int *lookuptable, /* The 3D look up table that will hold all the sum up values */
                 unsigned int *count,  /* The 3D count table containing all the times an entry was sumed up */
                 unsigned int lutsize /* The lookup table size */)
{
    unsigned int *row, *lutbaseaddr, *countbaseaddr;
    unsigned int index, index1, index2, index3, temp;
            
    int **datap = (int **)malloc(numrows * sizeof(int *)); /*Mapping 1D array to 2D*/
    for (index = 0; index < numrows; index++)
            datap[index] = data + index*numcols;

    unsigned int ***lookuptablep = (unsigned int ***)malloc(lutsize * sizeof(unsigned int **)); /*Mapping 1D arrays to 3D*/
    unsigned int ***countp = (unsigned int ***)malloc(lutsize * sizeof(unsigned int **));
    for (index = 0; index < lutsize; index++)
    {
        lookuptablep[index] = (unsigned int **)malloc(lutsize * sizeof(unsigned int *));
        countp[index] = (unsigned int **)malloc(lutsize * sizeof(int *));
        lutbaseaddr = (lookuptable + index*lutsize*lutsize);
        countbaseaddr = (count + index*lutsize*lutsize);
        
        for (index1 = 0; index1 < lutsize; index1++)
        {
            temp = index1*lutsize;
            lookuptablep[index][index1] = lutbaseaddr + temp;
            countp[index][index1] = countbaseaddr + temp;
        }
    }

    
    for (index = 0; index < numrows; index++)
    {
        row = datap[index];

        lookuptablep[row[0]][row[1]][row[2]] += row[3];
        if (isinf(lookuptablep[row[0]][row[1]][row[2]])
        {
           printf("Error, overflow detected! %f\n", lookuptablep[row[0]][row[1]][row[2]]);
           exit(-1);
        }

        countp[row[0]][row[1]][row[2]]++;
    }

    free(datap); /*Deallocating all pointers ...*/
    for (index = 0; index < lutsize; index++)
    {
        free(lookuptablep[index]);
        free(countp[index]);
    }
    
    free(lookuptablep);
    free(countp);
}

