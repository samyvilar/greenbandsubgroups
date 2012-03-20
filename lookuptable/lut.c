#include "lut.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* Author: Samy Vilar, create a lookup table on 4 bands, with first 3 being the indexes ...*/

void lookuptable(int *data,    /*2D DATA with n by r shape, n-1 used as indices, with DATA[n] being the sum up values.*/
                 unsigned int numrows, /* The number of rows withing the 2D data matrix. */
                 unsigned int numcols,  /* The number of columns withing the 2D data matrix */
                 float *lookuptable, /* The 3D look up table that will hold all the sum up values */
                 float *count,  /* The 3D count table containing all the times an entry was sumed up */
                 unsigned int lutsize /* The lookup table size */)
{
    int *row; float *lutbaseaddr, *countbaseaddr;
    unsigned int index, index1, temp;
            
    int **datap = (int **)malloc(numrows * sizeof(int *)); /*Mapping 1D array to 2D*/
    for (index = 0; index < numrows; index++)
            datap[index] = data + index*numcols;

    float ***lookuptablep = (float ***)malloc(lutsize * sizeof(float **)); /*Mapping 1D arrays to 3D*/
    float ***countp = (float ***)malloc(lutsize * sizeof(float **));
    for (index = 0; index < lutsize; index++)
    {
        lookuptablep[index] = (float **)malloc(lutsize * sizeof(float *));
        countp[index] = (float **)malloc(lutsize * sizeof(float *));
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
        if (isinf(lookuptablep[row[0]][row[1]][row[2]]))
        {
           printf("Error, overflow detected! %f\n", lookuptablep[row[0]][row[1]][row[2]]);
           exit(-1);
        }

        countp[row[0]][row[1]][row[2]] += 1;
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

