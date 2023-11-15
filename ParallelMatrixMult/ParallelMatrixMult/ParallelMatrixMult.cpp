#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>
#include <chrono>

using namespace std::chrono;


const int n = 2880;
/*const int block_size = 40;
const int block_count_in_row = n / block_size; 
const int block_count = (1 + block_count_in_row) * block_count_in_row / 2; 
const int block_elem_count = block_size * block_size;
const int block_arr_size = block_elem_count * block_count;*/

void fill_matrix_random_lower_triangle(int** matrix, int n,int maxValue)
{
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i+1; ++j) {
            matrix[i][j] = 1 + rand() % maxValue;
        }
    }
}

void fill_matrix_random_symmetric_upper_triangle(int** matrix, int n, int maxValue)
{
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            int elem = 1 + rand() % 16;
            matrix[i][j] = elem;
            matrix[j][i] = elem;
        }
    }
}


void printMatrix(int** matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%5d ", matrix[i][j]);
        }
        printf("\n");
    }
}

void fill_column_block_matrix_by_lower_triangle_matrix(int** lt_matrix, int* block_matrix, int n, int block_size)
{
    int ind = 0;
    for (int j = 0; j < n; j += block_size) {
        for (int i = j; i < n; i += block_size) {
            for (int k = 0; k < block_size; ++k) {
                for (int l = 0; l < block_size; ++l) {
                    block_matrix[ind] = lt_matrix[i + k][j + l];
                    ind++;
                }
            }
        }
    }
}

void fill_row_block_matrix_by_symmetric_upper_triangle_matrix(int** sym_matrix, int* block_matrix, int n, int block_size)
{
    int ind = 0;
    for (int i = 0; i < n; i += block_size) {
        for (int j = i; j < n; j += block_size) {
            for (int k = 0; k < block_size; ++k) {
                for (int l = 0; l < block_size; ++l) {
                    block_matrix[ind] = sym_matrix[i + k][j + l];
                    ind++;
                }
            }
        }
    }
}

//Перемножение матриц по определению
void default_matrix_multiplication(int** mat1, int** mat2, int** mat_res, int n)
{
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                mat_res[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
}


void mult(int* A, int* B, int** C, int numOfBlockRowA1, int numOfBlockRowA2, int numA1, int numA2, int block_size) {
    int block_elem_count = block_size * block_size;
    int startA1 = (numA1 - 1) * block_elem_count;
    int startA2 = (numA2 - 1) * block_elem_count;
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            int sum = 0;
            for (int k = 0; k < block_size; k++) {
                sum += A[startA1 + i * block_size + k] * B[startA2 + k * block_size + j];
            }
            C[block_size * (numOfBlockRowA1 - 1) + i][block_size * (numOfBlockRowA2 - 1) + j] += sum;
            //printf("%d %d \n",blockSize * (numOfBlockRowA1 - 1) + i , blockSize * (numOfBlockRowA2 - 1) + j);
        }
    }
}

void mult_parallel(int* A, int* B, int** C, int numOfBlockRowA1, int numOfBlockRowA2, int numA1, int numA2, int block_size) {
    int block_elem_count = block_size * block_size;
    int startA1 = (numA1 - 1) * block_elem_count;
    int startA2 = (numA2 - 1) * block_elem_count;
    #pragma omp parallel for
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            int sum = 0;
            for (int k = 0; k < block_size; k++) {
                sum += A[startA1 + i * block_size + k] * B[startA2 + k * block_size + j];
            }
            C[block_size * (numOfBlockRowA1 - 1) + i][block_size * (numOfBlockRowA2 - 1) + j] += sum;
            //printf("%d %d \n",blockSize * (numOfBlockRowA1 - 1) + i , blockSize * (numOfBlockRowA2 - 1) + j);
        }
    }
}

void mult_transpose(int* A, int* B, int** C, int numOfBlockRowA1, int numOfBlockRowA2, int numA1, int numA2, int block_size) {
    int block_elem_count = block_size * block_size;
    int startA1 = (numA1 - 1) * block_elem_count;
    int startA2 = (numA2 - 1) * block_elem_count;
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            int sum = 0;
            for (int k = 0; k < block_size; k++) {
                sum += A[startA1 + i * block_size + k] * B[startA2 + j * block_size + k];
            }
            C[block_size * (numOfBlockRowA1 - 1) + i][block_size * (numOfBlockRowA2 - 1) + j] += sum;
        }
    }
}

void mult_transpose_parallel(int* A, int* B, int** C, int numOfBlockRowA1, int numOfBlockRowA2, int numA1, int numA2, int block_size) {
    int block_elem_count = block_size * block_size;
    int startA1 = (numA1 - 1) * block_elem_count;
    int startA2 = (numA2 - 1) * block_elem_count;
    #pragma omp parallel for
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            int sum = 0;
            for (int k = 0; k < block_size; k++) {
                sum += A[startA1 + i * block_size + k] * B[startA2 + j * block_size + k];
            }
            C[block_size * (numOfBlockRowA1 - 1) + i][block_size * (numOfBlockRowA2 - 1) + j] += sum;
        }
    }
}

//Последовательное перемножение
void multiply_block_matrix(int* bm1, int* bm2, int** res_mat, int n, int block_size)
{
    int block_count_in_row = n / block_size;
    int block_count = (1 + block_count_in_row) * block_count_in_row / 2;

    int m1_cur_column = block_count_in_row - 1;
    int m1_cur_block = block_count-1;

    while (m1_cur_column >= 0)
    {
        //Поскольку m1 хранится по столбцам, а m2 по строкам, то номер 1-ого блока в перемножаемой строке
        //всегда будет соответствовать номеру первого блока в перемножаемом столбце
        int m2_notranspose_start_block = m1_cur_block;
        int m2_notranspose_end_block = m2_notranspose_start_block - ((block_count_in_row-1) - m1_cur_column);

        for (int m1_cur_row = block_count_in_row-1; m1_cur_row >= m1_cur_column; m1_cur_row--)
        {
            int m2_column = block_count_in_row;
            int m2_cur_block = m2_notranspose_start_block;
            for (; m2_cur_block >= m2_notranspose_end_block; --m2_cur_block)
            {
                --m2_column;
                //printf("%d * %d\n", m1_cur_block,m2_cur_block);
                //printf("row = %d col %d\n", m1_cur_row, m2_column);
                mult(bm1, bm2, res_mat, m1_cur_row+1, m2_column+1, m1_cur_block+1, m2_cur_block+1,block_size);
            }
            ++m2_cur_block;
            //поскольку после проведенных выше операций мы оказались в блоке слева от которого блоков нет(крайнем)
            //то для того чтобы перейти в верхний блок нам надо вычесть из текущего блока количество блоков, содержащихся в 
            //текущей строчке
            //когда мы поднимемся в самую верхнюю строчку, то это количество блоков станет равно количеству блоков в строчке
            //и соответственно выше нам подниматься не надо(да и не получится)
            for (int dec = block_count_in_row  - m1_cur_column; dec < block_count_in_row ; ++dec)
            {
                m2_cur_block -= dec;
                --m2_column;
                //printf(" transp: %d * %d\n", m1_cur_block, m2_cur_block);
                //printf("row = %d col %d\n", m1_cur_row, m2_column);
                mult_transpose(bm1, bm2, res_mat, m1_cur_row+1, m2_column+1, m1_cur_block+1, m2_cur_block+1,block_size);
            }


            //Поскольку мы обходим 1-ую матрицу по столбцам, то для того, чтобы поддерживать
            //номер блока актуальным достаточно просто его уменьшать на 1 после каждого умножения на столбец m2 
            m1_cur_block--;
        }

        m1_cur_column--;
    }
}
//Перемножение матриц с паралелльным перемножением внутри блока
void multiply_block_parallel_matrix(int* bm1, int* bm2, int** res_mat, int n, int block_size)
{
    int block_count_in_row = n / block_size;
    int block_count = (1 + block_count_in_row) * block_count_in_row / 2;

    int m1_cur_column = block_count_in_row - 1;
    int m1_cur_block = block_count - 1;

    while (m1_cur_column >= 0)
    {
        //Поскольку m1 хранится по столбцам, а m2 по строкам, то номер 1-ого блока в перемножаемой строке
        //всегда будет соответствовать номеру первого блока в перемножаемом столбце
        int m2_notranspose_start_block = m1_cur_block;
        int m2_notranspose_end_block = m2_notranspose_start_block - ((block_count_in_row - 1) - m1_cur_column);

        for (int m1_cur_row = block_count_in_row - 1; m1_cur_row >= m1_cur_column; m1_cur_row--)
        {
            int m2_column = block_count_in_row;
            int m2_cur_block = m2_notranspose_start_block;
            for (; m2_cur_block >= m2_notranspose_end_block; --m2_cur_block)
            {
                --m2_column;
                //printf("%d * %d\n", m1_cur_block,m2_cur_block);
                //printf("row = %d col %d\n", m1_cur_row, m2_column);
                mult_parallel(bm1, bm2, res_mat, m1_cur_row + 1, m2_column + 1, m1_cur_block + 1, m2_cur_block + 1, block_size);
            }
            ++m2_cur_block;
            //поскольку после проведенных выше операций мы оказались в блоке слева от которого блоков нет(крайнем)
            //то для того чтобы перейти в верхний блок нам надо вычесть из текущего блока количество блоков, содержащихся в 
            //текущей строчке
            //когда мы поднимемся в самую верхнюю строчку, то это количество блоков станет равно количеству блоков в строчке
            //и соответственно выше нам подниматься не надо(да и не получится)
            for (int dec = block_count_in_row - m1_cur_column; dec < block_count_in_row; ++dec)
            {
                m2_cur_block -= dec;
                --m2_column;
                //printf(" transp: %d * %d\n", m1_cur_block, m2_cur_block);
                //printf("row = %d col %d\n", m1_cur_row, m2_column);
                mult_transpose_parallel(bm1, bm2, res_mat, m1_cur_row + 1, m2_column + 1, m1_cur_block + 1, m2_cur_block + 1, block_size);
            }


            //Поскольку мы обходим 1-ую матрицу по столбцам, то для того, чтобы поддерживать
            //номер блока актуальным достаточно просто его уменьшать на 1 после каждого умножения на столбец m2 
            m1_cur_block--;
        }

        m1_cur_column--;
    }
}
//Перемножение матриц с паралелльным перемножением блоков
void multiply_block_matrix_parallel(int* bm1, int* bm2, int** res_mat,int n, int block_size)
{
    int block_count_in_row = n / block_size;
    int block_count = (1 + block_count_in_row) * block_count_in_row / 2;

    int m1_cur_column = block_count_in_row - 1;
    int m1_cur_block = block_count - 1;
    
    for (int m1_cur_column = block_count_in_row - 1; m1_cur_column >= 0; --m1_cur_column)
    {
        int m1_cur_block_new = m1_cur_block - (block_count_in_row - m1_cur_column);
        //printf("%d",m1_cur_block);
        //Поскольку m1 хранится по столбцам, а m2 по строкам, то номер 1-ого блока в перемножаемой строке
        //всегда будет соответствовать номеру первого блока в перемножаемом столбце
        int m2_notranspose_start_block = m1_cur_block;
        int m2_notranspose_end_block = m2_notranspose_start_block - ((block_count_in_row - 1) - m1_cur_column);
        #pragma omp parallel for
        for (int m1cb = m1_cur_block; m1cb >m1_cur_block_new; m1cb--)
        {
            //printf("%d\n", omp_get_thread_num());
            int m1_cur_row = block_count_in_row - 1 - (m1_cur_block - m1cb);

            int m2_column = block_count_in_row;
            int m2_cur_block = m2_notranspose_start_block;
            for (; m2_cur_block >= m2_notranspose_end_block; --m2_cur_block)
            {
                --m2_column;
                //printf("%d * %d\n", m1cb,m2_cur_block);
                //printf("row = %d col %d\n", m1_cur_row, m2_column);
                mult(bm1, bm2, res_mat, m1_cur_row + 1, m2_column + 1, m1cb + 1, m2_cur_block + 1,block_size);
            }
            ++m2_cur_block;
            //поскольку после проведенных выше операций мы оказались в блоке слева от которого блоков нет(крайнем)
            //то для того чтобы перейти в верхний блок нам надо вычесть из текущего блока количество блоков, содержащихся в 
            //текущей строчке
            //когда мы поднимемся в самую верхнюю строчку, то это количество блоков станет равно количеству блоков в строчке
            //и соответственно выше нам подниматься не надо(да и не получится)
            for (int dec = block_count_in_row - m1_cur_column; dec < block_count_in_row; ++dec)
            {
                m2_cur_block -= dec;
                --m2_column;
                //printf(" transp: %d * %d\n", m1cb, m2_cur_block);
                //printf("row = %d col %d\n", m1_cur_row, m2_column);
                mult_transpose(bm1, bm2, res_mat, m1_cur_row + 1, m2_column + 1, m1cb + 1, m2_cur_block + 1,block_size);
            }
                


            //Поскольку мы обходим 1-ую матрицу по столбцам, то для того, чтобы поддерживать
            //номер блока актуальным достаточно просто его уменьшать на 1 после каждого умножения на столбец m2 
            //m1_cur_block--;

        }
        m1_cur_block = m1_cur_block_new;
       
    }
}
//Проверка 2-х матриц на эквивалентность
bool matrix_equal(int** m1, int** m2, int n)
{
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (m1[i][j] != m2[i][j])
            {
                return false;
            }
        }
    }
    return true;
}

bool nullify_matrix(int** m, int n)
{
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            m[i][j] = 0;
        }
    }
    return true;
}


int main() {
    srand(time(NULL));

    int blockSizes[] = { 4, 8,12, 16, 20, 40, 45,60, 80, 120, 160,180,240,360,480};
    //int blockSizes[] = { 2,4, 8,  16, 32,64 };
    int** first = (int**)calloc(n, sizeof(int*));
    for (int i = 0; i < n; i++) {
        first[i] = (int*)calloc(n, sizeof(int));
    }
    int** second = (int**)calloc(n, sizeof(int*));
    for (int i = 0; i < n; i++) {
        second[i] = (int*)calloc(n, sizeof(int));
    }

    omp_set_num_threads(6);

    fill_matrix_random_lower_triangle(first, 15, n);

    fill_matrix_random_symmetric_upper_triangle(second, 15, n);


    int** res_mat = (int**)calloc(n, sizeof(int*));
    for (int i = 0; i < n; i++) {
        res_mat[i] = (int*)calloc(n, sizeof(int));
    }
    int** res_verification_matrix = (int**)calloc(n, sizeof(int*));
    for (int i = 0; i < n; i++) {
        res_verification_matrix[i] = (int*)calloc(n, sizeof(int));
    }
    printf("Default matrix multiplication calculation...\n");
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    default_matrix_multiplication(first, second, res_verification_matrix, n);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    long double duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    printf("Calculation time = %Lf ms\n", duration);
    for (size_t i = 0; i < 15; i++)
    {
        int block_size = blockSizes[i];
        int block_count_in_row = n / block_size;
        int block_count = (1 + block_count_in_row) * block_count_in_row / 2;

        int* bFirst = (int*)calloc(block_size*block_size * block_count, sizeof(int));
        int* bSecond = (int*)calloc(block_size * block_size * block_count, sizeof(int));

        printf("Block size = %d\n", block_size);
        fill_column_block_matrix_by_lower_triangle_matrix(first, bFirst, n, block_size);

        fill_row_block_matrix_by_symmetric_upper_triangle_matrix(second, bSecond, n, block_size);

        printf("Sequential calculation...\n");
        t1 = high_resolution_clock::now();
        multiply_block_matrix(bFirst, bSecond, res_mat, n, block_size);
        t2 = high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        bool equal = matrix_equal(res_mat, res_verification_matrix, n);
        printf("Calculation accuracy = %d\n", equal);
        printf("Calculation time = %Lf ms\n", duration);

        nullify_matrix(res_mat,n);

        printf("Parallel in block calculation...\n");
        t1 = high_resolution_clock::now();
        multiply_block_parallel_matrix(bFirst, bSecond, res_mat, n, block_size);
        t2 = high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        equal = matrix_equal(res_mat, res_verification_matrix, n);
        printf("Calculation accuracy = %d\n", equal);
        printf("Calculation time = %Lf ms\n", duration);

        nullify_matrix(res_mat, n);

        printf("Parallel between blocks calculation...\n");
        t1 = high_resolution_clock::now();
        multiply_block_matrix_parallel(bFirst, bSecond, res_mat, n, block_size);
        t2 = high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        equal = matrix_equal(res_mat, res_verification_matrix, n);
        printf("Calculation accuracy = %d\n", equal);
        printf("Calculation time = %Lf ms\n", duration);

        nullify_matrix(res_mat, n);

        free(bFirst);
        free(bSecond);
    }

   
}