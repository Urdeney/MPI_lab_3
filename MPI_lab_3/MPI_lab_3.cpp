#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include "mpi.h"



using namespace std;
using namespace std::chrono;

// Условия выполнения программы: все матрицы квадратные,
// размер блоков и их количество по горизонтали и вертикали
// одинаково, процессы образуют квадратную решетку
int ProcNum = 0;  // Количество доступных процессов
int ProcRank = 0;  // Ранг текущего процесса
int GridSize;    // Размер виртуальной решетки процессов
int GridCoords[2]; // Координаты текущего процесса в процессной
          // решетке
MPI_Comm GridComm; // Коммуникатор в виде квадратной решетки
MPI_Comm ColComm;  // коммуникатор – столбец решетки
MPI_Comm RowComm;  // коммуникатор – строка решетки



void PrintMatrix(double* Matrix, int size)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            cout << setw(4) << Matrix[i * size + j] << "|";
        }
        cout << endl;
    }
}


// Создание коммуникатора в виде двумерной квадратной решетки
// и коммуникаторов для каждой строки и каждого столбца решетки
void CreateGridCommunicators()
{
    int DimSize[2];   // Количество процессов в каждом измерении
              // решетки
    int Periodic[2];  // =1 для каждого измерения, являющегося
              // периодическим
    int Subdims[2];   // =1 для каждого измерения, оставляемого
              // в подрешетке
    DimSize[0] = GridSize;
    DimSize[1] = GridSize;
    Periodic[0] = 0;
    Periodic[1] = 0;
    // Создание коммуникатора в виде квадратной решетки
    MPI_Cart_create(MPI_COMM_WORLD, 2, DimSize, Periodic, 1, &GridComm);
    // Определение координат процесса в решетке
    MPI_Cart_coords(GridComm, ProcRank, 2, GridCoords);
    // Создание коммуникаторов для строк процессной решетки
    Subdims[0] = 0; // Фиксация измерения
    Subdims[1] = 1; //Наличие данного измерения в подрешетке
    MPI_Cart_sub(GridComm, Subdims, &RowComm);
    // Создание коммуникаторов для столбцов процессной решетки
    Subdims[0] = 1;
    Subdims[1] = 0;
    MPI_Cart_sub(GridComm, Subdims, &ColComm);
}

// Генерация рандомной матрицы
void RandomDataInitialization(double* pAMatrix, double* pBMatrix, int Size) {
    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < Size * Size; i++) {
        pAMatrix[i] = rand() % 10;
        pBMatrix[i] = rand() % 10;
    }
}

// Функция для выделения памяти и инициализации исходных данных
void ProcessInitialization(double*& pAMatrix, double*& pBMatrix,
    double*& pCMatrix, double*& pAblock, double*& pBblock,
    double*& pCblock, double*& pTemporaryAblock, int& Size,
    int& BlockSize)
{
    if (ProcRank == 0)
    {
        do
        {
            cout<<"Enter matrix size" << endl;
            scanf("%d", &Size);

            if (Size % GridSize != 0)
            {
                cout << "Number of processes must be a perfect square!" << endl;
            }
        } while (Size % GridSize != 0);
    }
    MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    BlockSize = Size / GridSize;
    pAblock = new double[BlockSize * BlockSize];
    pBblock = new double[BlockSize * BlockSize];
    pCblock = new double[BlockSize * BlockSize];
    pTemporaryAblock = new double[BlockSize * BlockSize];
    for (int i = 0; i < BlockSize * BlockSize; i++)
    {
        pCblock[i] = 0;
    }
    if (ProcRank == 0)
    {
        pAMatrix = new double[Size * Size];
        pBMatrix = new double[Size * Size];
        pCMatrix = new double[Size * Size];
        //RandomDataInitialization(pAMatrix, pBMatrix, Size);
        for (int i = 0; i < Size * Size; i++) {
            pAMatrix[i] = i; // Случайные числа от 0 до 9
            pBMatrix[i] = i;
        }
        cout << "Matrix A" << endl;
        PrintMatrix(pAMatrix, Size);
        cout << "Matrix B" << endl;
        PrintMatrix(pBMatrix, Size);
    }
}

// Рассылка блоков матрицы А по строкам решетки процессов
void ABlockCommunication(int iter, double* pAblock,
    double* pMatrixAblock, int BlockSize)
{
    // Определение ведущего процесса в строке процессной решетки
    int Pivot = (GridCoords[0] + iter) % GridSize;

    // Копирование передаваемого блока в отдельный буфер памяти
    if (GridCoords[1] == Pivot)
    {
        for (int i = 0; i < BlockSize * BlockSize; i++)
            pAblock[i] = pMatrixAblock[i];
    }
    // Рассылка блока
    MPI_Bcast(pAblock, BlockSize * BlockSize, MPI_DOUBLE, Pivot, RowComm);
}

// Умножение матричных блоков
void BlockMultiplication(double* pAblock, double* pBblock,
    double* pCblock, int BlockSize)
{
    // Вычисление произведения матричных блоков
    for (int i = 0; i < BlockSize; i++)
    {
        for (int j = 0; j < BlockSize; j++)
        {
            double temp = 0;
            for (int k = 0; k < BlockSize; k++)
                temp += pAblock[i * BlockSize + k] * pBblock[k * BlockSize + j];
            pCblock[i * BlockSize + j] += temp;
        }
    }
}

// Циклический сдвиг блоков матрицы В вдоль столбца процессной решетки
void BblockCommunication(double* pBblock, int BlockSize)
{
    MPI_Status Status;
    int NextProc = GridCoords[0] + 1;
    if (GridCoords[0] == GridSize - 1)
        NextProc = 0;
    int PrevProc = GridCoords[0] - 1;
    if (GridCoords[0] == 0)
        PrevProc = GridSize - 1;
    MPI_Sendrecv_replace(pBblock, BlockSize * BlockSize, MPI_DOUBLE,
        NextProc, 0, PrevProc, 0, ColComm, &Status);
}

// Функция для параллельного умножения матриц
void ParallelResultCalculation(double* pAblock, double* pMatrixAblock,
    double* pBblock, double* pCblock, int BlockSize)
{
    for (int iter = 0; iter < GridSize; iter++)
    {
        // Рассылка блоков матрицы A по строкам процессной решетки
        ABlockCommunication(iter, pAblock, pMatrixAblock, BlockSize);
        // Умножение блоков
        BlockMultiplication(pAblock, pBblock, pCblock, BlockSize);
        // Циклический сдвиг блоков матрицы B в столбцах процессной решетки
        BblockCommunication(pBblock, BlockSize);
    }
}

// Распределние исходных данных
void DataDistribution(double* pAMatrix, double* pBMatrix, double* pMatrixAblock, double* pBblock, int Size, int BlockSize) {
    MPI_Datatype BlockType, ResizedBlockType;
    /*
    MPI_Type_vector(BlockSize, BlockSize, Size, MPI_DOUBLE, &BlockType);
    MPI_Type_create_resized(BlockType, 0, BlockSize * sizeof(double), &ResizedBlockType);
    MPI_Type_commit(&ResizedBlockType);
    */

    int* SendCounts = new int[ProcNum];
    int* Displs = new int[ProcNum];

    if (ProcRank == 0) {
        int Disp = 0;
        for (int i = 0; i < GridSize; i++) {
            for (int j = 0; j < GridSize; j++) {
                SendCounts[i * GridSize + j] = 1;
                Displs[i * GridSize + j] = Disp;
                Disp += 1;
            }
            Disp += (BlockSize - 1) * GridSize;
        }
    }

    MPI_Scatterv(pAMatrix, SendCounts, Displs, ResizedBlockType, pMatrixAblock, BlockSize * BlockSize, MPI_DOUBLE, 0, GridComm);
    MPI_Scatterv(pBMatrix, SendCounts, Displs, ResizedBlockType, pBblock, BlockSize * BlockSize, MPI_DOUBLE, 0, GridComm);

    delete[] SendCounts;
    delete[] Displs;
    //MPI_Type_free(&ResizedBlockType);
}

//Сбор результирующей матрицы
void ResultCollection(double* pCMatrix, double* pCblock, int Size, int BlockSize) {
    MPI_Datatype BlockType, ResizedBlockType;
    /*
    MPI_Type_vector(BlockSize, BlockSize, Size, MPI_DOUBLE, &BlockType);
    MPI_Type_create_resized(BlockType, 0, BlockSize * sizeof(double), &ResizedBlockType);
    MPI_Type_commit(&ResizedBlockType);
    */

    int* SendCounts = new int[ProcNum];
    int* Displs = new int[ProcNum];

    if (ProcRank == 0) {
        int Disp = 0;
        for (int i = 0; i < GridSize; i++) {
            for (int j = 0; j < GridSize; j++) {
                SendCounts[i * GridSize + j] = 1;
                Displs[i * GridSize + j] = Disp;
                Disp += 1;
            }
            Disp += (BlockSize - 1) * GridSize;
        }
    }

    MPI_Gatherv(pCblock, BlockSize * BlockSize, MPI_DOUBLE, pCMatrix, SendCounts, Displs, ResizedBlockType, 0, GridComm);

    delete[] SendCounts;
    delete[] Displs;
    //MPI_Type_free(&ResizedBlockType);
}


//Очистка памяти
void ProcessTermination(double* pAMatrix, double* pBMatrix, double* pCMatrix, double* pAblock, double* pBblock, double* pCblock, double* pMatrixAblock) {
    if (ProcRank == 0) {
        delete[] pAMatrix;
        delete[] pBMatrix;
        delete[] pCMatrix;
    }
    delete[] pAblock;
    delete[] pBblock;
    delete[] pCblock;
    delete[] pMatrixAblock;

    _CrtDumpMemoryLeaks();
}


void main(int argc, char* argv[])
{
    setlocale(LC_ALL, "Russian");

    double* pAMatrix;  // Первый аргумент матричного умножения
    double* pBMatrix;  // Второй аргумент матричного умножения
    double* pCMatrix;  // Результирующая матрица
    int Size;      // Размер матриц
    int BlockSize;   // Размер матричных блоков, расположенных
              // на процессах
    double* pAblock;  // Блок матрицы А на процессе
    double* pBblock;  // Блок матрицы В на процессе
    double* pCblock;  // Блок результирующей матрицы С на процессе
    double* pMatrixAblock;
    double Start, Finish, Duration;
    setvbuf(stdout, 0, _IONBF, 0);
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    GridSize = sqrt((double)ProcNum);
    if (ProcNum != GridSize * GridSize)
    {
        if (ProcRank == 0)
        {
            cout << "Number of processes must be a perfect square" << endl;
        }
    }
    else
    {
        if (ProcRank == 0)
            cout << "Parallel matrix multiplication program" << endl;

        // Создание виртуальной решетки процессов и коммуникаторов строк и столбцов
        CreateGridCommunicators();
        // Выделение памяти и инициализация элементов матриц
        ProcessInitialization(pAMatrix, pBMatrix, pCMatrix, pAblock, pBblock, pCblock, pMatrixAblock, Size, BlockSize);
        // Блочное распределение матриц между процессами
        DataDistribution(pAMatrix, pBMatrix, pMatrixAblock, pBblock, Size,
            BlockSize);
        // Выполнение параллельного метода Фокса
        ParallelResultCalculation(pAblock, pMatrixAblock, pBblock,
            pCblock, BlockSize);
        // Сбор результирующей матрицы на ведущем процессе
        ResultCollection(pCMatrix, pCblock, Size, BlockSize);

        if (ProcRank == 0) {
            cout << "Result matrix" << endl;
            PrintMatrix(pCMatrix, Size);
        }

        // Завершение процесса вычислений
        ProcessTermination(pAMatrix, pBMatrix, pCMatrix, pAblock, pBblock, pCblock, pMatrixAblock);
    }
    MPI_Finalize();
    
}