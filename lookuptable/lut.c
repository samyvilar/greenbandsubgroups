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

void update_min_max_lut(float *prev_values,
                        float *new_values,
                        unsigned int lut_size,
                        unsigned int function)
{
    unsigned int index = 0, index_1 = 0, index_2 = 0, temp = 0;

    //map_1d_array_to_3d(prev_values, lut_size);
    //map_1d_array_to_3d(new_values, lut_size);
    float *prev_values_base_address = NULL, *new_values_base_address = NULL;

    float ***prev_values_p = (float ***)malloc(lut_size * sizeof(float **)); /*Mapping 1D arrays to 3D*/
    float ***new_values_p = (float ***)malloc(lut_size * sizeof(float **));
    for (index = 0; index < lut_size; index++)
    {
        prev_values_p[index] = (float **)malloc(lut_size * sizeof(float *));
        prev_values_base_address = (prev_values + index*lut_size*lut_size);

        new_values_p[index] = (float **)malloc(lut_size * sizeof(float *));
        new_values_base_address = (new_values + index*lut_size*lut_size);

        for (index_1 = 0; index_1 < lut_size; index_1++)
        {
            temp = index_1*lut_size;
            prev_values_p[index][index_1] = prev_values_base_address + temp;
            new_values_p[index][index_1] = new_values_base_address + temp;
        }
    }



    for (index = 0; index < lut_size; index++)
        for (index_1 = 0; index_1 < lut_size; index_1++)
            for (index_2 = 0; index_2 < lut_size; index_2++)
                if (function == 0)
                {
                    if (new_values_p[index][index_1][index_2] < prev_values_p[index][index_1][index_2])
                    {   prev_values_p[index][index_1][index_2] = new_values_p[index][index_1][index_2]; }
                }
                else if (function == 1)
                {
                    if (new_values_p[index][index_1][index_2] >= prev_values_p[index][index_1][index_2])
                    {   prev_values_p[index][index_1][index_2] = new_values_p[index][index_1][index_2]; }
                }
                else
                {
                    printf("Only supporting min(0) max (1) got %i \n", function);
                    exit(-2);
                }

}

void set_min_max     (int         *data,
                      unsigned int numrows,
                      unsigned int numcols,
                      float        *lookuptable,
                      unsigned int lutsize,
                      unsigned int function)
{
    int *row = NULL; float *lutbaseaddr = NULL;
    unsigned int index = 0, index1 = 0, temp = 0;

    int **datap = (int **)malloc(numrows * sizeof(int *)); /*Mapping 1D array to 2D*/
    for (index = 0; index < numrows; index++)
            datap[index] = data + index*numcols;

    float ***lookuptablep = (float ***)malloc(lutsize * sizeof(float **)); /*Mapping 1D arrays to 3D*/
    for (index = 0; index < lutsize; index++)
    {
        lookuptablep[index] = (float **)malloc(lutsize * sizeof(float *));
        lutbaseaddr = (lookuptable + index*lutsize*lutsize);

        for (index1 = 0; index1 < lutsize; index1++)
        {
            temp = index1*lutsize;
            lookuptablep[index][index1] = lutbaseaddr + temp;
        }
    }


    for (index = 0; index < numrows; index++)
    {
        row = datap[index];
        if (function == 0)
        {
            if (row[3] < lookuptablep[row[0]][row[1]][row[2]])
            {    lookuptablep[row[0]][row[1]][row[2]] = row[3]; }
        }
        else if (function == 1)
        {
            if (row[3] >= lookuptablep[row[0]][row[1]][row[2]])
            {    lookuptablep[row[0]][row[1]][row[2]] = row[3]; }
        }
        else
        {
            printf("Error! only supporting 0 for min and 1 for max got %i\n", function);
            exit(-1);
        }
    }

    free(datap); /*Deallocating all pointers ...*/
    for (index = 0; index < lutsize; index++)
        free(lookuptablep[index]);

    free(lookuptablep);
}



void lookuptable(int          *data,        /* 2D DATA with n by r shape, n-1 used as indices, with DATA[n] being the sum up values.*/
                 unsigned int numrows,      /* The number of rows withing the 2D data matrix. */
                 unsigned int numcols,      /* The number of columns withing the 2D data matrix */
                 float        *lookuptable, /* The 3D look up table that will hold all the sum up values */
                 float        *count,       /* The 3D count table containing all the times an entry was sum up */
                 unsigned int lutsize      /* The lookup table size */)
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
                        double          *lookuptable_flatten,
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
                if (lookuptablep[index][index1][index2])
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

