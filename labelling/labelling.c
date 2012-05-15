
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "labelling.h"

void set_labels(double *data, unsigned int data_number_of_rows, unsigned int data_number_of_columns,
               double *means, unsigned int number_of_sub_groups, unsigned int means_number_of_rows, unsigned int means_number_of_columns,
               unsigned int *labels)
{
    unsigned int index = 0, index_1 = 0, index_2 = 0, index_3 = 0;
    double **datap = (double **)malloc(data_number_of_rows * sizeof(double *)); /*Mapping 1D array to 2D*/
    for (index = 0; index < data_number_of_rows; index++)
        datap[index] = data + index*data_number_of_columns;

    unsigned int size = 0, size_1 = 0;
    if (number_of_sub_groups == 1)
    {
        double **meansp = (double **)malloc(means_number_of_rows * sizeof(double *));
        for (index = 0; index < means_number_of_rows; index++)
            meansp[index] = means + index*means_number_of_columns;

        double diff = 0;
        //dist = numpy.zeros((data.shape[0], means.shape[0]))
        //for i in xrange(means.shape[0]):
        //    dist[:, i] = numpy.sum((data - means[i,:])**2, axis = 1)
        double **distances = (double **)malloc(data_number_of_rows * sizeof(double *));
        size = means_number_of_columns * sizeof(double);
        for (index = 0; index < data_number_of_rows; index++)
        {
            distances[index] = (double *)malloc(size);
            memset(distances[index], 0, size);
        }

         /* Calculate the distances to each group for each row by squaring the distance to each center
          * for each group, and taking that sum.
          */
        printf("calculating distances \n");
        for (index = 0; index < means_number_of_rows; index++)
            for (index_1 = 0; index_1 < data_number_of_rows; index_1++)
                for (index_2 = 0; index_2 < means_number_of_columns; index_2++)
                {
                    diff = (datap[index_1][index_2] - meansp[index][index_2]);
                    distances[index_1][index] += (diff * diff);
                }
        printf("done\nLabelling...\n");
        for (index = 0; index < data_number_of_rows; index++)
        {
            labels[index] = 0;
            for (index_1 = 0; index_1 < means_number_of_rows; index_1++)
                if (distances[index][index_1] < distances[index][labels[index]])
                    labels[index] = index_1;
        }
        printf("done\n");

        printf("tying to free\n");
        for (index = 0; index < data_number_of_rows; index++)
            free(distances[index]);
        printf("freeing distances done.\n");
        free(distances);
        printf("free distances dine.\n");
        free(meansp);
        printf("free meansp done\n");
    }
    else
    {
        double ***meansp = (double ***)malloc(number_of_sub_groups * sizeof(double **));
        double *baseaddress = NULL;
        size = means_number_of_rows * sizeof(double *);
        for (index = 0; index < number_of_sub_groups; index++)
        {
            meansp[index] = (double **)malloc(size);
            baseaddress = means + index*number_of_sub_groups*means_number_of_rows;
            for (index_1 = 0; index_1 < means_number_of_rows; index_1++)
                meansp[index][index_1] = baseaddress + index_1*means_number_of_columns;
        }

        //dist = numpy.zeros((data.shape[0], means.shape[0], means.shape[1]))
        double ***distances = (double ***)malloc(data_number_of_rows * sizeof(double **));
        size = number_of_sub_groups * sizeof(double *);
        size_1 = means_number_of_columns * sizeof(double);
        for (index = 0; index < data_number_of_rows; index++)
        {
            distances[index] = (double **)malloc(size);
            for (index_1 = 0; index_1 < means_number_of_rows; index_1++)
            {
                distances[index][index_1] = (double *)malloc(size_1);
                memset(distances[index][index_1], 0, size_1);
            }
        }
        double **total_distances = (double **)malloc(data_number_of_rows * sizeof(double *));
        size = number_of_sub_groups * sizeof(double);
        for (index = 0; index < data_number_of_rows; index++)
        {
            total_distances[index] = (double *)malloc(size);
            memset(total_distances[index], 0, size);
        }

        double diff = 0;
        for (index = 0; index < number_of_sub_groups; index++)
            for (index_1 = 0; index_1 < means_number_of_rows; index_1++)
                for (index_2 = 0; index_2 < data_number_of_rows; index_2++)
                {
                    for (index_3 = 0; index_3 < means_number_of_columns; index_3++)
                    {
                        diff = (datap[index_2][index_3] - meansp[index][index_1][index_3]);
                        distances[index_2][index][index_1] += (diff * diff);
                    }
                    total_distances[index_2][index] += distances[index_2][index][index_1];
                }

        unsigned int **labelsp = (unsigned int **)malloc(data_number_of_rows * sizeof(unsigned int *));
        for (index = 0; index < data_number_of_rows; index++)
            labelsp[index] = labels + index * 2;

        for (index = 0; index < data_number_of_rows; index++)
        {
            labelsp[index][0] = 0;
            for (index_1 = 0; index_1 < number_of_sub_groups; index_1++)
                if (total_distances[index][index_1] < total_distances[index][labelsp[index][0]])
                    labelsp[index][0] = index_1;

            labelsp[index][1] = 0;
            for (index_1 = 0; index_1 < means_number_of_rows; index_1++)
                if (distances[index][labelsp[index][0]][index_1] < distances[index][labelsp[index][0]][labelsp[index][1]])
                    labelsp[index][1] = index_1;
        }

        /*\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ */
        for (index = 0; index < number_of_sub_groups; index++)
        {
            free(meansp[index]);
            free(total_distances[index]);
        }
        free(meansp);
        free(total_distances);
        for (index = 0; index < data_number_of_rows; index++)
        {
            for (index_1 = 0; index_1 < means_number_of_rows; index_1++)
                free(distances[index][index_1]);
            free(distances[index]);
        }
        free(distances);
        free(labelsp);
    }


    free(datap);
}

