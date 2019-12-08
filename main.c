#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>
#include <memory.h>

static const long Num_To_Sort = 1000000000;

//Quick Sort algorithm from https://www.programiz.com/dsa/quick-sort
//Swaps the places of two values in the array
void swap(int *a, int *b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

//Divides and sorts subdivisions
int partition(int array[], int low, int high)
{
    int pivot = array[high];
    int i = (low - 1);
    for (int j = low; j < high; j++)
    {
        if (array[j] <= pivot)
        {
            i++;
            swap(&array[i], &array[j]);
        }
    }
    swap(&array[i + 1], &array[high]);
    return (i + 1);
}

//Serial quick sort algorithm
void quickSortSerial(int array[], int low, int high)
{
    if (low < high)
    {
        int pi = partition(array, low, high);
        quickSortSerial(array, low, pi - 1);
        quickSortSerial(array, pi + 1, high);
    }
}

//Parallel quick sort algorithm
//Learned about pragma omp task from https://stackoverflow.com/questions/40961392/best-way-to-parallelize-this-recursion-using-openmp
void quickSortParallel(int array[], int low, int high)
{
    if (low < high)
    {
        int pi = partition(array, low, high);
        #pragma omp task //Defines block of code to be executed in parallel that can handle irregular algorithms such as recursion
        {
            quickSortSerial(array, low, pi - 1);
        }
        #pragma omp task //Defines block of code to be executed in parallel that can handle irregular algorithms such as recursion
        {
            quickSortSerial(array, pi + 1, high);
        }
    }
}

// Sequential version of your sort
// If you're implementing the PSRS algorithm, you may ignore this section
void sort_s(int *arr) {
    quickSortSerial(arr, 0, Num_To_Sort - 1);
}

// Parallel version of your sort
//Use of pragma omp single also from https://stackoverflow.com/questions/40961392/best-way-to-parallelize-this-recursion-using-openmp
void sort_p(int *arr) {
    #pragma omp parallel num_threads(omp_get_max_threads()) //Defines block of code to be executed in parallel
    {
        #pragma omp single //Used to specify a block of code that must be run by a single thread
        {
            quickSortParallel(arr, 0, Num_To_Sort - 1);
        }
    }
}

int main() {
    int *arr_s = malloc(sizeof(int) * Num_To_Sort);
    long chunk_size = Num_To_Sort / omp_get_max_threads();
    #pragma omp parallel num_threads(omp_get_max_threads())
    {
        int p = omp_get_thread_num();
        unsigned int seed = (unsigned int) time(NULL) + (unsigned int) p;
        long chunk_start = p * chunk_size;
        long chunk_end = chunk_start + chunk_size;
        for (long i = chunk_start; i < chunk_end; i++) {
            arr_s[i] = rand_r(&seed);
        }
    }

    // Copy the array so that the sorting function can operate on it directly.
    // Note that this doubles the memory usage.
    // You may wish to test with slightly smaller arrays if you're running out of memory.
    int *arr_p = malloc(sizeof(int) * Num_To_Sort);
    memcpy(arr_p, arr_s, sizeof(int) * Num_To_Sort);

    struct timeval start, end;

    printf("Timing sequential...\n");
    gettimeofday(&start, NULL);
    sort_s(arr_s);
    gettimeofday(&end, NULL);
    printf("Took %f seconds\n\n", end.tv_sec - start.tv_sec + (double) (end.tv_usec - start.tv_usec) / 1000000);

    free(arr_s);

    printf("Timing parallel...\n");
    gettimeofday(&start, NULL);
    sort_p(arr_p);
    gettimeofday(&end, NULL);
    printf("Took %f seconds\n\n", end.tv_sec - start.tv_sec + (double) (end.tv_usec - start.tv_usec) / 1000000);

    free(arr_p);

    return 0;
}
