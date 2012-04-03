#include "lut.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* Author: Samy Vilar, create a lookup table on 4 bands, with first 3 being the indexes ...*/

void lookuptable(int          *data,        /* 2D DATA with n by r shape, n-1 used as indices, with DATA[n] being the sum up values.*/
                 unsigned int numrows,      /* The number of rows withing the 2D data matrix. */
                 unsigned int numcols,      /* The number of columns withing the 2D data matrix */
                 float        *lookuptable, /* The 3D look up table that will hold all the sum up values */
                 float        *count,       /* The 3D count table containing all the times an entry was sum up */
                 unsigned int lutsize       /* The lookup table size */)
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

void predict_double(int             *data,
                    unsigned int    numrows,
                    unsigned int    numcols,
                    double          *lookuptable,
                    unsigned int    lutsize,
                    double          *results)
{
    int *row; double *lutbaseaddr;
    unsigned int index, index1;


    int **datap = (int **)malloc(numrows * sizeof(int *)); /*Mapping 1D array to 2D*/
    for (index = 0; index < numrows; index++)
        datap[index] = data + index*numcols;

    double ***lookuptablep = (double ***)malloc(lutsize * sizeof(double **)); /*Mapping 1D arrays to 3D*/
    for (index = 0; index < lutsize; index++)
    {
        lookuptablep[index] = (double **)malloc(lutsize * sizeof(double *));
        lutbaseaddr = (lookuptable + index*lutsize*lutsize);

        for (index1 = 0; index1 < lutsize; index1++)
            lookuptablep[index][index1] = lutbaseaddr + index1*lutsize;
    }


    for (index = 0; index < numrows; index++)
    {
        row    = datap[index];
        if (row[0] > lutsize || row[1] > lutsize || row[2] > lutsize)
        {
            printf("Error: Index exceeding look up table size %i\n", lutsize);
            exit(-1);
        }
        results[index] = lookuptablep[row[0]][row[1]][row[2]];
    }
}

void flatten_lookuptable(double         *lookuptable,
                        unsigned int    lutsize,
                        double          *lookuptable_flatten
                        unsigned int    numrows)
{
    unsigned int index, index1, index2;
    double *lutbaseaddr;
    double ***lookuptablep = (double ***)malloc(lutsize * sizeof(double **)); /*Mapping 1D arrays to 3D*/

    double **lookuptable_flattenp = (double **)malloc(numrows * sizeof(double *)); /*Mapping 1D array to 2D*/
    for (index = 0; index < numrows; index++)
        lookuptable_flattenp[index] = lookuptable_flatten + index*4;

    for (index = 0; index < lutsize; index++)
    {
        lookuptablep[index] = (double **)malloc(lutsize * sizeof(double *));
        lutbaseaddr = (lookuptable + index*lutsize*lutsize);

        for (index1 = 0; index1 < lutsize; index1++)
            lookuptablep[index][index1] = lutbaseaddr + index1*lutsize;
    }

    unsigned int row = 0;
    for (index = 0; index < lutsize; index++)
        for (index1 = 0; index1 < lutsize; index1++)
            for (index2 = 0; index2 < lutsize; index2++)
                if (lookuptable[index][index1][index2])
                {
                    if (row > numrows)
                    {
                        printf("Error: Number of rows exceeded set max %i\n", numrows);
                        exit(-1);
                    }
                    lookuptable_flattenp[row][0] = (double)index;
                    lookuptable_flattenp[row][1] = (double)index1;
                    lookuptable_flattenp[row][2] = (double)index2;
                    lookuptable_flattenp[row][3] = lookuptablep[index][index1][index2];
                    row++;
                }

}

