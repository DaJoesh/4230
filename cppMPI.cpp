#include <stdio.h>
#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <ctime>

#include <mpi.h>

using namespace std;

// Definitions
typedef std::chrono::duration<double> Runtime; // One of many possible ways to record the time

// Prototypes
void printMatrix(const int* A, const int m, const int n);
int* randomMatrix(const int m, const int n);
int compareMatrix(const int* A, const int* B, const int m, const int n);
int* MM_sequential(const int* A, const int* B, const int m, const int n, const int q);
int* MM_1D_Distributed(const int* A, const int* B, const int m, const int n, const int q);

int main(int argc, char* argv[]) {
    // ------ VARIABLE SETUPS ------
    int pid; // Processor ID
    int p_total; // Number of Processors Being Used

    // Formatting timers
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();

    // --------------- BEGIN MPI ---------------
    MPI_Init(NULL, NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &p_total);

    // User input for matrix
    int m = 16;
    int n = 16;
    int q = 16;

    if (argc >= 4) {
        m = stoi(argv[1]);
        n = stoi(argv[2]);
        q = stoi(argv[3]);
    }

    // If a dimension is not divisible by processors
    if (m % p_total != 0 || n % p_total != 0 || q % p_total != 0) {
        if (pid == 0) {
            cout << "Matrix dimensions are not divisible by the number of processors.\n";
        }
        MPI_Finalize();
        return 1;
    }

    // MASTER THREAD
    if (pid == 0) {
        // Code for error broadcasting
        int broadcastCode = 0;

        // Attempt to open the file
        std::ofstream outputfile("output.txt");

        // Failed to open file
        if (!outputfile.is_open()) {
            std::cerr << "Error opening file." << std::endl;
            broadcastCode = 1;
            for (int i = 1; i < p_total; i++) {
                MPI_Send(&broadcastCode, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }

            MPI_Finalize();
            return 1;
        } else {
            for (int i = 1; i < p_total; i++) {
                MPI_Send(&broadcastCode, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }

        // HEADERS
        outputfile << std::left
                   << std::setw(10) << "m"
                   << std::setw(10) << "n"
                   << std::setw(10) << "q"
                   << std::setw(10) << "p"
                   << std::setw(25) << "Time_Sequential"
                   << std::setw(25) << "Time_1D"
                   << std::setw(25) << "Time_2D"
                   << std::endl;

        outputfile << std::left
                   << std::setw(10) << m
                   << std::setw(10) << n
                   << std::setw(10) << q
                   << std::setw(10) << p_total;


        int* A_matrix;
        int* B_matrix;
        int* sequential_matrix;
        int* parallel_matrix;

        // Two same sized matrices
        A_matrix = randomMatrix(m, n);
        B_matrix = randomMatrix(n, q);

        // ------ SEQUENTIAL ------
        start = std::chrono::high_resolution_clock::now();
        sequential_matrix = MM_sequential(A_matrix, B_matrix, m, n, q);
        end = std::chrono::high_resolution_clock::now();
        Runtime sequential_time = end - start;

        outputfile << std::left << std::setw(25) << sequential_time.count();

        // ------ 1D ALGORITHM ------
        start = std::chrono::high_resolution_clock::now();
        parallel_matrix = MM_1D_Distributed(A_matrix, B_matrix, m, n, q);
        end = std::chrono::high_resolution_clock::now();
        Runtime oneDimension_time = end - start;

        // Check Correctness
        if (compareMatrix(sequential_matrix, parallel_matrix, m, q) == 1) {
            outputfile << std::left << std::setw(25) << "ERROR FOUND";
            broadcastCode = 1;
            for (int i = 1; i < p_total; i++) {
                MPI_Send(&broadcastCode, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }

            MPI_Finalize();
            return 1;
        } else {
            outputfile << std::left << std::setw(25) << oneDimension_time.count();
            for (int i = 1; i < p_total; i++) {
                MPI_Send(&broadcastCode, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }

        // ------ 2D ALGORITHM ------
        start = std::chrono::high_resolution_clock::now();

        // TODO
        end = std::chrono::high_resolution_clock::now();
        Runtime twoDimension_time = end - start;

        // Check Correctness
        if (compareMatrix(sequential_matrix, parallel_matrix, m, q) == 1) {
            outputfile << std::left << std::setw(25) << "ERROR FOUND";
            broadcastCode = 1;
            for (int i = 1; i < p_total; i++) {
                MPI_Send(&broadcastCode, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }

            MPI_Finalize();
            return 1;
        } else {
            outputfile << std::left << std::setw(25) << twoDimension_time.count();
            for (int i = 1; i < p_total; i++) {
                MPI_Send(&broadcastCode, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }

        outputfile.close();
        free(A_matrix);
        free(B_matrix);
        free(sequential_matrix);
        free(parallel_matrix);

    } else {
        int number;
        MPI_Status status;
        // Look for error from open file
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        if (number == 1) {
            MPI_Finalize();
            return 0;
        }
	}

	MPI_Finalize();
	return 0;
}
