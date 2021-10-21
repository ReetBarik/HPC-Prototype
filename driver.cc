#include <iostream>
#include "omp.h"
#include "driver.h"

// nvcc -x cu -Xcompiler -fopenmp -o driver driver.cc
// ./driver 4 2

int main(int argc, char *argv[]) {

	int dimension1, num_gpu_workers; 

	dimension1 = atoi(argv[1]);
	num_gpu_workers = atoi(argv[2]);

	int size = 50;

	int *array1, *array2, *array3;

	array1 = (int*)malloc(size * sizeof(int));
	array2 = (int*)malloc(size * sizeof(int));
	array3 = (int*)malloc(size * sizeof(int));

	for (int i = 0; i < size; i++) {
		array1[i] = i;
		array2[i] = i; 
	}

	driver(array1, array2, array3, size, dimension1, num_gpu_workers);

	for (int i = 0; i < size; i++) {
		std::cout << array3[i] << " "; 
	}
	std::cout << std::endl;

	return 0;
}