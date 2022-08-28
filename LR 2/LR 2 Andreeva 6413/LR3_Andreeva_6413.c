#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include < math.h >
#include <stdbool.h>

//параллельный алгоритм openMPI
double** allocMatrix(int rows, int column);
void createMatrix(double** A, int n, int m);
void printMatrix(double** A, int n, int m);


int main(int argc, char** argv) {
    int n = atoi(argv[argc - 1]);
    int m = n;
    double eps0 = 0.001;
    int rank, size;
    MPI_Status status;
    double** A = allocMatrix(n, m);
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double duration = MPI_Wtime();
    createMatrix(A, n, m);
    double Anew;
    double toComp;
    if (rank == 0) {
        double eps1 = 0;
        //Первый поток рассчитывает первую половины матрицы
        for (int i = 1; i < n / 2; i++) {
            for (int j = 1; j < m - 1; j++) {
                Anew = (A[i + 1][j] + A[i - 1][j] + A[i][j + 1] + A[i][j - 1]) / 4;
                if (fabs(Anew) > 1) {
                    toComp = fabs(Anew - A[i][j]) / Anew;
                }
                else {
                    toComp = fabs(Anew - A[i][j]);
                }
                if (toComp > eps1) {
                    eps1 = toComp;
                }
                A[i][j] = Anew;
            }
        }
        //Отправляем последнюю строку первой половины матрицы
        MPI_Send(&(A[n / 2 - 1][0]), m, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
        //Отправляем точность на первой половине матрицы
        MPI_Send(&(eps1), 1, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);
    }
    if (rank == 0) {
        double fl = false;
        while (!fl)
        {
            double eps1 = 0;
            //Первый поток рассчитывает первую половины матрицы кроме последней строки блока
            for (int i = 1; i < n / 2 - 1; i++) {
                for (int j = 1; j < m - 1; j++) {
                    Anew = (A[i + 1][j] + A[i - 1][j] + A[i][j + 1] + A[i][j - 1]) / 4;
                    if (fabs(Anew) > 1) {
                        toComp = fabs(Anew - A[i][j]) / Anew;
                    }
                    else {
                        toComp = fabs(Anew - A[i][j]);
                    }
                    if (toComp > eps1) {
                        eps1 = toComp;
                    }
                    A[i][j] = Anew;
                }
            }
            //Получаем первую строку второй половины матрицы
            MPI_Recv(&(A[n / 2][0]), m, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
            //Первый поток рассчитывает последнюю строку первой половины матрицы
            for (int j = 1; j < m - 1; j++) {
                Anew = (A[n / 2 - 1 + 1][j] + A[n / 2 - 1 - 1][j] + A[n / 2 - 1][j + 1] + A[n / 2 - 1][j - 1]) / 4;
                if (fabs(Anew) > 1) {
                    toComp = fabs(Anew - A[n / 2 - 1][j]) / Anew;
                }
                else {
                    toComp = fabs(Anew - A[n / 2 - 1][j]);
                }
                if (toComp > eps1) {
                    eps1 = toComp;
                }
                A[n / 2 - 1][j] = Anew;
            }
            //Получаем значение флага о достижении заданной точности
            MPI_Recv(&(fl), 1, MPI_C_BOOL, 1, 3, MPI_COMM_WORLD, &status);
            //Отправляем последнюю строку первой половины матрицы
            MPI_Send(&(A[n / 2 - 1][0]), m, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
            //Отправляем точность на первой половине матрицы
            MPI_Send(&(eps1), 1, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);
        }
    }
    if (rank == 1) {
        bool fl = false;
        do {
            double eps = 0;
            double eps2 = 0;
            //Получаем последнюю строку первой половины матрицы
            MPI_Recv(&(A[n / 2 - 1][0]), m, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
            //Получаем точность на первой половине матрицы
            MPI_Recv(&(eps), m, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &status);
            //Рассчитывается первая строка второй половины матрицы
            for (int j = 1; j < m - 1; j++) {
                Anew = (A[n / 2 + 1][j] + A[n / 2 - 1][j] + A[n / 2][j + 1] + A[n / 2][j - 1]) / 4;
                if (fabs(Anew) > 1) {
                    toComp = fabs(Anew - A[n / 2][j]) / Anew;
                }
                else {
                    toComp = fabs(Anew - A[n / 2][j]);
                }
                if (toComp > eps2) {
                    eps2 = toComp;
                }
                A[n / 2][j] = Anew;
            }
            //Отправляем первую строку второй половины матрицы
            MPI_Send(&(A[n / 2][0]), m, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            //Второй поток продолжает рассчитывать вторую половину матрицы
            for (int i = n / 2 + 1; i < n - 1; i++) {
                for (int j = 1; j < m - 1; j++) {
                    Anew = (A[i + 1][j] + A[i - 1][j] + A[i][j + 1] + A[i][j - 1]) / 4;
                    if (fabs(Anew) > 1) {
                        toComp = fabs(Anew - A[i][j]) / Anew;
                    }
                    else {
                        toComp = fabs(Anew - A[i][j]);
                    }
                    if (toComp > eps2) {
                        eps2 = toComp;
                    }
                    A[i][j] = Anew;
                }
            }
            eps = max(eps, eps2);
            //printf("eps: %f\n", eps);
            if (eps < eps0) {
                fl = true;
            }
            //Отправляем значение флага о достижении заданной точности
            MPI_Send(&(fl), 1, MPI_C_BOOL, 0, 3, MPI_COMM_WORLD);
        } while (!fl);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        duration = MPI_Wtime() - duration;
        printf("Duration: %f\n", duration);
    }
    free(A[0]);
    free(A);
    MPI_Finalize();
    return 0;
}
//создаем матрицу
double** allocMatrix(int rows, int column) {
    double* data = (double*)malloc(rows * column * sizeof(double));
    double** matr = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++)
        matr[i] = &(data[column * i]);
    return matr;
}
//заполняем матрицу значениями
void createMatrix(double** A, int n, int m) {
    double t0 = 5;
    double t1 = 85;
    double l = (t1 - t0) / (n - 1);
    for (int i = 0; i < m; i++) {
        A[0][i] = t1;
        A[n - 1][i] = t0;
    }
    for (int i = 0; i < n; i++) {
        A[i][0] = t1 - l * i;
        A[i][m - 1] = t1 - l * i;
    }
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < m - 1; j++)
        {
            A[i][j] = 0.;
        }
    }
}

void printMatrix(double** A, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%.1f ", A[i][j]);
        }
        printf("\n");
    }
}

/*
//параллельный алгоритм openMP
double** allocMatrix(int rows, int column);
void createMatrix(double** A, int n, int m);
void main() {
    int n, m;
    printf("Insert n: ");
    scanf_s("%d", &n);
    m = n;
    double eps0 = 0.001;
    double** A = allocMatrix(n, m);
    clock_t start = clock();
    createMatrix(A, n, m);
    double eps = 0;
    double Anew;
    double toComp;
    bool dataForFirstThreadIsReady = false;
    bool dataForSecondThreadIsReady = false;
    bool fl = false;
    bool epsIsActual = false;
    //Рассчитывается первая половина матрицы
    for (int i = 1; i < m / 2; i++) {
        for (int j = 1; j < m - 1; j++) {
            Anew = (A[i + 1][j] + A[i - 1][j] + A[i][j + 1] + A[i][j - 1]) / 4;
            if (fabs(Anew) > 1) {
                toComp = fabs(Anew - A[i][j]) / Anew;
            }
            else {
                toComp = fabs(Anew - A[i][j]);
            }
            if (toComp > eps) {
                eps = toComp;
            }
            A[i][j] = Anew;
        }
    }
    epsIsActual = true;
    //Второй поток может приступать к расчету первой строки второй половины матрицы
    dataForSecondThreadIsReady = true;
    do {
#pragma omp parallel sections num_threads(8) shared(A, dataForFirstThreadIsReady, dataForSecondThreadIsReady, fl, eps, epsIsActual) private(Anew, toComp)
        {
            //Рассчитывается вторая половина матрицы
#pragma omp section
            {
                double eps2 = 0;
                //Ждем, пока первый поток вычислит последнюю строчку
                while (!dataForSecondThreadIsReady) { printf("."); };
                //Рассчитывается первая строка второй половины матрицы
                for (int j = 1; j < m - 1; j++) {
                    Anew = (A[n / 2 + 1][j] + A[n / 2 - 1][j] + A[n / 2][j + 1] + A[n / 2][j - 1]) / 4;
                    if (fabs(Anew) > 1) {
                        toComp = fabs(Anew - A[n / 2][j]) / Anew;
                    }
                    else {
                        toComp = fabs(Anew - A[n / 2][j]);
                    }
                    if (toComp > eps2) {
                        eps2 = toComp;
                    }
                    A[n / 2][j] = Anew;
                }
                dataForSecondThreadIsReady = false;
                //Первый поток теперь может рассчитать последнюю строку первой половины матрицы
                dataForFirstThreadIsReady = true;
                //Второй поток продолжает рассчитывать вторую половину матрицы
                for (int i = n / 2 + 1; i < n - 1; i++) {
                    for (int j = 1; j < m - 1; j++) {
                        Anew = (A[i + 1][j] + A[i - 1][j] + A[i][j + 1] + A[i][j - 1]) / 4;
                        if (fabs(Anew) > 1) {
                            toComp = fabs(Anew - A[i][j]) / Anew;
                        }
                        else {
                            toComp = fabs(Anew - A[i][j]);
                        }
                        if (toComp > eps2) {
                            eps2 = toComp;
                        }
                        A[i][j] = Anew;
                    }
                }
                while (!epsIsActual) {};
                eps = max(eps, eps2);
                //printf("eps: %f\n", eps);
                if (eps < eps0) {
                    fl = true;
                }
                eps = 0;
                epsIsActual = false;
            }
            //Рассчитывается первая половина матрицы
#pragma omp section
            {
                double eps1 = 0;
                //Первый поток рассчитывает первую половины матрицы кроме последней строки блока
                for (int i = 1; i < n / 2 - 1; i++) {
                    for (int j = 1; j < m - 1; j++) {
                        Anew = (A[i + 1][j] + A[i - 1][j] + A[i][j + 1] + A[i][j - 1]) / 4;
                        if (fabs(Anew) > 1) {
                            toComp = fabs(Anew - A[i][j]) / Anew;
                        }
                        else {
                            toComp = fabs(Anew - A[i][j]);
                        }
                        if (toComp > eps1) {
                            eps1 = toComp;
                        }
                        A[i][j] = Anew;
                    }
                }
                //Ждем, когда второй поток посчитает первую строку, чтобы первый мог приступить к рассчету последней
                while (!dataForFirstThreadIsReady) { printf("."); };
                //Первый поток рассчитывает последнюю строку первой половины матрицы
                for (int j = 1; j < m - 1; j++) {
                    Anew = (A[n / 2 - 1 + 1][j] + A[n / 2 - 1 - 1][j] + A[n / 2 - 1][j + 1] + A[n / 2 - 1][j - 1]) / 4;
                    if (fabs(Anew) > 1) {
                        toComp = fabs(Anew - A[n / 2 - 1][j]) / Anew;
                    }
                    else {
                        toComp = fabs(Anew - A[n / 2 - 1][j]);
                    }
                    if (toComp > eps1) {
                        eps1 = toComp;
                    }
                    A[n / 2 - 1][j] = Anew;
                }
                dataForFirstThreadIsReady = false;
                //Теперь второй поток может вычислять первую строку второй половины матрицы
                dataForSecondThreadIsReady = true;
                while (epsIsActual) {};
                eps = eps1;
                epsIsActual = true;
            }
        }
    } while (!fl);

    clock_t stop = clock();
    double duration = (double)(stop - start) / CLOCKS_PER_SEC;
    printf("Duration: %f sec\n", duration);
    free(A[0]);
    free(A);
}

//создаем матрицу
double** allocMatrix(int rows, int column) {
    double* data = (double*)malloc(rows * column * sizeof(double));
    double** matr = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++)
        matr[i] = &(data[column * i]);
    return matr;
}
//заполняем матрицу значениями
void createMatrix(double** A, int n, int m) {
    double t0 = 5;
    double t1 = 85;
    double l = (t1 - t0) / (n - 1);
    for (int i = 0; i < m; i++) {
        A[0][i] = t1;
        A[n - 1][i] = t0;
    }
    for (int i = 0; i < n; i++) {
        A[i][0] = t1 - l * i;
        A[i][m - 1] = t1 - l * i;
    }
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < m - 1; j++)
        {
            A[i][j] = 0.;
        }
    }
}
*/

/*
//последовательный алгоритм
double** allocMatrix(int rows, int column);
void createMatrix(double** A, int n, int m);
void main() {
    int n, m;
    printf("Insert n: ");
    scanf_s("%d", &n);
    m = n;
    double eps0 = 0.001;
    double** A = allocMatrix(n, m);
    createMatrix(A, n, m);
    double eps;
    double Anew;
    double toComp;
    clock_t start = clock();
    do {
        eps = 0;
        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < m - 1; j++) {
                Anew = (A[i + 1][j] + A[i - 1][j] + A[i][j + 1] + A[i][j - 1]) / 4;
                if (fabs(Anew) > 1) {
                    toComp = fabs(Anew - A[i][j]) / Anew;
                }
                else {
                    toComp = fabs(Anew - A[i][j]);
                }
                if (toComp > eps) {
                    eps = toComp;
                }
                A[i][j] = Anew;
            }
        }
    } while (eps0 < eps);
    clock_t stop = clock();
    double duration = (double)(stop - start) / CLOCKS_PER_SEC;
    printf("Duration: %f sec\n", duration);
    free(A[0]);
    free(A);
}

//создаем матрицу
double** allocMatrix(int rows, int column) {
    double* data = (double*)malloc(rows * column * sizeof(double));
    double** matr = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++)
        matr[i] = &(data[column * i]);
    return matr;
}
//заполняем матрицу значениями
void createMatrix(double** A, int n, int m) {
    double t0 = 5;
    double t1 = 85;
    double l = (t1 - t0) / (n - 1);
    for (int i = 0; i < m; i++) {
        A[0][i] = t1;
        A[n - 1][i] = t0;
    }
    for (int i = 0; i < n; i++) {
        A[i][0] = t1 - l * i;
        A[i][m - 1] = t1 - l * i;
    }
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < m - 1; j++)
        {
            A[i][j] = 0.;
        }
    }
}
*/