#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "mpi.h"

int main(int argc, char** argv) {

	int n = 1000000000;

	int rank, size;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int n1 = n / size;

	double duration = MPI_Wtime();
	MPI_Bcast(&n1, 1, MPI_INT, 0, MPI_COMM_WORLD);
	int m = 0;
	for (int i = 0; i < n1; i++) {
		double x = (double)rand() / (double)RAND_MAX;
		double dgr = ((double)rand() / (double)RAND_MAX) * M_PI;
		if (x < 0.5 * sin(dgr) || (1 - x) < 0.5 * sin(dgr)) {
			m++;
		}
	}
	int m_fin = 0;
	MPI_Reduce(&m, &m_fin, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		double pi = 2 * (double)n / (double)m_fin;
		printf("n = %d\npi = %.12f\n", n, pi);
		duration = MPI_Wtime() - duration;
		printf("t = %f", duration);
	}
	MPI_Finalize();
	return 0;

}



/*
//параллельный алгоритм openMP
void main() {
	int n;
	double pi;
	int m = 0, i;
	double t1, t2;
	printf("Enter n = ");
	scanf_s("%d", &n);

	t1 = omp_get_wtime();//время начала работы алгоритма 
	m = 0;


#pragma omp parallel for private(i) reduction(+ : m) shared(n) num_threads(8) 

	for (i = 0; i < n; i++) 
	{
		double x = (double)rand() / (double)RAND_MAX;
		double dgr = ((double)rand() / (double)RAND_MAX) * M_PI;
		if (x < 0.5 * sin(dgr) || (1 - x) < 0.5 * sin(dgr)) {
			m++;
		}
	}

	pi = 2 * (double)n / (double)m;
	printf("pi = %.12f\n", pi);
	t2 = omp_get_wtime();//время окончания работы алгоритма 
	printf("t = %f\n", t2 - t1);
	
}
*/

/* //последовательный алгоритм
void main() {
	int n;
	int m = 0;
	double t1, t2;
	printf("Enter n = ");
	scanf_s("%d", &n);

	t1 = omp_get_wtime();//время начала работы алгоритма
	for (int i = 0; i < n; i++) {
		double x = (double)rand() / (double)RAND_MAX;
		double dgr = ((double)rand() / (double)RAND_MAX) * M_PI;
		if (x < 0.5 * sin(dgr) || (1 - x) < 0.5 * sin(dgr)) {
			m++;
		}
	}
	double pi = 2 * (double)n / (double)m;
	printf("pi = %.12f\n", pi);
	t2 = omp_get_wtime();//время окончания работы алгоритма
	m = 0;
	printf("t = %f\n", t2 - t1);
}
*/
