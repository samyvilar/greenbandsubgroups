#include "lut.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* Author: Samy Vilar, create a lookup table on 4 bands, with first 3 being the indexes ...*/

float ***map_1d_array_to_3d(float *values, unsigned int lut_size)
{
    unsigned int index = 0, index_1 = 0, temp = 0;
    float *values_base_addr;
    float ***values_p = (float ***)malloc(lut_size * sizeof(float **)); /*Mapping 1D arrays to 3D*/
    for (index = 0; index < lut_size; index++)
        {
            values_p[index] = (float **)malloc(lut_size * sizeof(float *));
            values_base_addr = (values + index*lut_size*lut_size);

            for (index_1 = 0; index_1 < lut_size; index_1++)
            {
                temp = index_1*lut_size;
                values_p[index][index_1] = values_base_addr + temp;
            }
        }
    return values_p;
}

void free_3d_array(float ***values, unsigned int lut_size)
{
    unsigned int index = 0;
    for (index = 0; index < lut_size; index++)
        free(values[index]);

    free(values);
}


void set_min_max     (int         *data,
                      unsigned int numrows,
                      unsigned int numcols,
                      float        *mins,
                      float        *max,
                      unsigned int lutsize)
{
    int *row = NULL; float *minsp_base_address = NULL, *maxp_base_addrress = NULL;
    unsigned int index = 0, index1 = 0, temp = 0;

    int **datap = (int **)malloc(numrows * sizeof(int *)); /*Mapping 1D array to 2D*/
    for (index = 0; index < numrows; index++)
        datap[index] = data + index*numcols;

    float ***minsp = malloc(lutsize * sizeof(float **)); /*Mapping 1D arrays to 3D*/
    float ***maxp = malloc(lutsize * sizeof(float **));

    for (index = 0; index < lutsize; index++)
    {
        minsp[index] = malloc(lutsize * sizeof(float *));
        maxp[index] = malloc(lutsize * sizeof(float *));

        minsp_base_address = (mins + index*lutsize*lutsize);
        maxp_base_addrress = (max + index*lutsize*lutsize);

        for (index1 = 0; index1 < lutsize; index1++)
        {
            temp = index1*lutsize;
            minsp[index][index1] = minsp_base_address + temp;
            maxp[index][index1] = maxp_base_addrress + temp;
        }
    }


    for (index = 0; index < numrows; index++)
    {
        row = datap[index];

        if (minsp[row[0]][row[1]][row[2]] >= row[3])
            minsp[row[0]][row[1]][row[2]] = row[3];

        if (maxp[row[0]][row[1]][row[2]] <= row[3])
            maxp[row[0]][row[1]][row[2]] = row[3];


    }

    free(datap); /*Deallocating all pointers ...*/
    for (index = 0; index < lutsize; index++)
    {
        free(minsp[index]);
        free(maxp[index]);
    }

    free(minsp);
    free(maxp);
}



void lookuptable(int          *data,        /* 2D DATA with n by r shape, n-1 used as indices, with DATA[n] being the sum up values.*/
                 unsigned int numrows,      /* The number of rows withing the 2D data matrix. */
                 unsigned int numcols,      /* The number of columns withing the 2D data matrix */
                 unsigned int *lookuptable, /* The 3D look up table that will hold all the sum up values */
                 unsigned int *count,       /* The 3D count table containing all the times an entry was sum up */
                 unsigned int lutsize      /* The lookup table size */
                 )
{
    int *row; unsigned int *lut_base_addr, *count_base_addr;
    unsigned int index, index1, temp;
            
    int **data_p = malloc(numrows * sizeof(int *)); /*Mapping 1D array to 2D*/
    for (index = 0; index < numrows; index++)
        data_p[index] = data + index*numcols;

    unsigned int ***lookuptable_p = malloc(lutsize * sizeof(unsigned int **)); /*Mapping 1D arrays to 3D*/
    unsigned int ***count_p = malloc(lutsize * sizeof(unsigned int **));

    for (index = 0; index < lutsize; index++)
    {
        lookuptable_p[index] = malloc(lutsize * sizeof(unsigned int *));
        count_p[index] = malloc(lutsize * sizeof(unsigned int *));

        long long int temp_base_addr_offset = index*lutsize*lutsize;
        lut_base_addr = lookuptable + temp_base_addr_offset;
        count_base_addr = (count + temp_base_addr_offset);

        for (index1 = 0; index1 < lutsize; index1++)
        {
            temp = index1*lutsize;
            lookuptable_p[index][index1] = lut_base_addr + temp;
            count_p[index][index1] = count_base_addr + temp;
        }
    }
    
    for (index = 0; index < numrows; index++)
    {
        row = data_p[index];

        lookuptable_p[row[0]][row[1]][row[2]] += row[3];

        if (lookuptable_p[row[0]][row[1]][row[2]] < 0)
        {
           printf("Error, overflow detected! %u\n", lookuptable_p[row[0]][row[1]][row[2]]);
           exit(-1);
        }

        count_p[row[0]][row[1]][row[2]] += 1;
        if (count_p[row[0]][row[1]][row[2]] < 0)
        {
            printf("Error, overflow detected! %u\n", count_p[row[0]][row[1]][row[2]]);
            exit(-1);
        }
    }

    free(data_p); /*Deallocating all pointers ...*/
    for (index = 0; index < lutsize; index++)
    {
        free(lookuptable_p[index]);
        free(count_p[index]);
    }
    
    free(lookuptable_p);
    free(count_p);
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
                        double          *lookuptable_flatten,
                        unsigned int    numrows,
                        int default_value)
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
                if (lookuptablep[index][index1][index2] != default_value)
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

