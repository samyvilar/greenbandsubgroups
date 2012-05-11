
#include <stdlib.h>
#include <string.h>

#include "labelling.h"

void set_labels(double *data, unsigned int data_number_of_rows, unsigned int data_number_of_columns,
               double *means, unsigned int number_of_sub_groups, unsigned int means_number_of_rows, unsigned int means_number_or_columns,
               unsigned int *labels)
{
    unsigned int index = 0, index_1 = 0, index_2 = 0;
    double **datap = (double **)malloc(data_number_of_rows * sizeof(double *)); /*Mapping 1D array to 2D*/
    for (index = 0; index < data_number_of_rows; index++)
        datap[index] = data + index*data_number_of_columns;

    if (number_of_sub_groups == 1)
    {
        double **meansp = (double **)malloc(means_number_of_rows * sizeof(double *));
        double diff = 0;
        //dist = numpy.zeros((data.shape[0], means.shape[0]))
        //for i in xrange(means.shape[0]):
        //    dist[:, i] = numpy.sum((data - means[i,:])**2, axis = 1)
        double **distances = (double **)malloc(data_number_of_rows * sizeof(double *));
        for (index = 0; index < data_number_of_rows; index++)
        {
            distances[index] = (double *)malloc(means_number_or_columns * sizeof(double));
            memset(distances[index], 0, means_number_or_columns * sizeof(double));
        }

        for (index = 0; index < means_number_of_rows; index++)
            meansp[index] = means + index*means_number_or_columns;

         /* Calculate the distances to each group for each row by squaring the distance to each center
          * for each group, and taking that sum.
          */
        for (index = 0; index < means_number_of_rows; index++)
            for (index_1 = 0; index_1 < data_number_of_rows; index_1++)
                for (index_2 = 0; index_2 < means_number_or_columns; index_2++)
                {
                    diff = (datap[index_1][index_2] - meansp[index][index_2]);
                    distances[index_1][index] += diff * diff;
                }
        for (index = 0; index < data_number_of_rows; index++)
        {
            labels[index] = 0;
            for (index_1 = 0; index_1 < means_number_of_rows; index_1++)
                if (distances[index][index_1] < distances[index][labels[index]])
                    labels[index] = index_1;
        }
        for (index = 0; index < data_number_of_rows; index++)
            free(distances[index]);
        free(distances);
        free(meansp);
    }
    else
    {
        double ***meansp = (double ***)malloc(number_of_sub_groups * sizeof(double **));
        double *baseaddress = NULL;
        for (index = 0; index < number_of_sub_groups; index++)
        {
            meansp[index] = (double **)malloc(means_number_of_rows * sizeof(double *));
            baseaddress = means + index*number_of_sub_groups*means_number_of_rows;
            for (index_1 = 0; index_1 < means_number_of_rows; index_1++)
                meansp[index][index_1] = baseaddress + index_1*means_number_or_columns;
        }
    }


    free(datap);


}

