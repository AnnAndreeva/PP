#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <malloc.h>

#include <omp.h>
#include "mpi.h"

//#define N 2000

////параллельный алгоритм с MPI
int main(int* argc, char** argv)
{
	const int N = 2000;
	double t1, t2;

	int size, rank;
	MPI_Status status;
	MPI_Init(argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int chunk = N / size;
	double* matrA = (double*)malloc(sizeof(double) * N * N);

	if (rank == 0) {
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				matrA[j + i * N] = 1.0;
	}

	double* matrB = (double*)malloc(sizeof(double) * N * N);

	if (rank == 0) {
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				matrB[j + i * N] = 1.0;
	}

	double* PartRes = (double*)malloc(sizeof(double) * chunk * N);
	double* matrC = (double*)malloc(sizeof(double) * N * N);

	t1 = MPI_Wtime();//время начала работы алгоритма 

	if (rank == 0) {

		for (int i = 1; i < size; i++) {
			//передача сообщений между процессами
			MPI_Send(matrA + i * N * chunk, N * chunk, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		}
	}
	else
	{
		//прием сообщения процессом получателем
		MPI_Recv(matrA, N * chunk, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	}

	//осуществление рассылки всем процессам
	MPI_Bcast(matrB, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	for (int i = 0; i < chunk; i++) {
		for (int j = 0; j < N; j++) {
			PartRes[i * N + j] = 0;
			for (int k = 0; k < N; k++)
				PartRes[i * N + j] += matrA[i * N + k] * matrB[k * N + j];
		}
	}

	if (rank != 0) {
		MPI_Send(PartRes, N * chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}

	if (rank == 0) {
		for (int i = 0; i < chunk * N; i++) {
			matrC[i] = PartRes[i];
		}

		for (int i = 1; i < size; i++) {
			MPI_Recv(matrC + i * N * chunk, N * chunk, MPI_DOUBLE, i,
				MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		}
	}

	t2 = MPI_Wtime();//время окончания работы алгоритма

	if (rank == 0) {
		printf("N=%d\n", N);
		printf("threads=%d\n", size);
		printf("t= %f\n", t2 - t1);
	}
	MPI_Finalize();
	system("pause");
}

/* //параллельный алгоритм с OpenMP
int i, j, k;
int main() {

	omp_set_num_threads(8);//количество процессов
	double t1, t2;

	double** matrA = (double**)malloc(N * sizeof(double*));
	double** matrB = (double**)malloc(N * sizeof(double*));
	double** matrC = (double**)malloc(N * sizeof(double*));
	
	for (int i = 0; i < N; i++) {
		matrA[i] = (double*)malloc(N * sizeof(double));
		matrB[i] = (double*)malloc(N * sizeof(double));
		matrC[i] = (double*)malloc(N * sizeof(double));
	}

	srand(1);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
		{
			matrA[i][j] = rand();
			matrB[i][j] = rand();
		}
	}
	
	t1 = omp_get_wtime();//время начала работы алгоритма 

 #pragma omp parallel for private(i,j,k) shared(matrA,matrB,matrC) schedule(dynamic) //распараллеленный алгоритм умножения матриц

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			matrC[i][j] = 0;
			for (k = 0; k < N; k++) {
				matrC[i][j] = matrC[i][j] + matrA[i][k] * matrB[k][j];
			}
		}
	}

	t2 = omp_get_wtime();//время окончания работы алгоритма
	printf("t = %f\n", t2 - t1);
	for (int i = 0; i < N; i++)
	{
		free(matrA[i]);
		free(matrB[i]);
		free(matrC[i]);
	}
	free(matrA);
	free(matrB);
	free(matrC);
	system("pause");
}
*/

/* //последовательный алгоритм
int main() {
	double t1, t2;

	double** matrA = (double**)malloc(N * sizeof(double*));
	double** matrB = (double**)malloc(N * sizeof(double*));
	double** matrC = (double**)malloc(N * sizeof(double*));

	for (int i = 0; i < N; i++) {
		matrA[i] = (double*)malloc(N * sizeof(double));
		matrB[i] = (double*)malloc(N * sizeof(double));
		matrC[i] = (double*)malloc(N * sizeof(double));
	}

	srand(1);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
		{
			matrA[i][j] = rand();
			matrB[i][j] = rand();;
		}
	}

	t1 = omp_get_wtime();//время начала работы алгоритма
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			matrC[i][j] = 0;
			for (int k = 0; k < N; k++) {
				matrC[i][j] += matrA[i][k] * matrB[k][j];
			}
		}
	}
	t2 = omp_get_wtime();//время окончания работы алгоритма

	printf("t = %f\n", t2 - t1);
	for (int i = 0; i < N; i++)
	{
		free(matrA[i]);
		free(matrB[i]);
		free(matrC[i]);
	}
	free(matrA);
	free(matrB);
	free(matrC);
	system("pause");
}
*/

