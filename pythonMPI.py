from mpi4py import MPI
import numpy as np
import time
import math

#  This method generates random data for matrices A and B and distributes it among the different processors. 
def distribute_data(n, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Calculate the number of rows each processor will receive
    local_n = n // size
    # Calulate the number of leftover rows
    remainder = n % size
    # Add one row to each processor until there are no more leftover rows
    local_n += 1 if rank < remainder else 0

    A = np.random.rand(local_n, n)
    B = np.random.rand(n, local_n)

    return A, B

# This method performs matrix multiplication. Pretty simple.
def matrix_multiply(A, B):
    return np.dot(A, B)

def main():
    # Sets up the MPI communicator object which represents all processors in the MPI world
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Values of n (matrix size) and p (number of processors)
    n_values = [100, 1000, 5000, 10000]
    # Look at size and if size is 8 then do 1, 2, 4, 8, size is 4 then do 1, 2, 4, size is 2 then do 1, 2, size is 1 then do 1
    p_values = [2**i for i in range(int(math.log2(size)), -1, -1)]

    for n in n_values:
        for p in p_values:
            # Start the timer on the root processor
            if rank == 0:
                start_time = time.time()

            # Distribute the data among the processors
            A, B = distribute_data(n, comm)

            # End the timer on the root processor
            if rank == 0:
                end_time = time.time()
                runtime = end_time - start_time
                print(f"\nRuntime for n={n}, p={p}: {runtime} seconds\n")
                
                # # If p = 1, store the runtime as the sequential time as there is no parallel time to compare to
                # if p <= 1:
                #     sequential_time = runtime
                    
                #     # Calculate the speedup for every other processor
                # elif p > 1:
                #     parallel_time = runtime
                #     speedup = sequential_time / parallel_time
                #     print(f"\n-----------Speedup for n={n}, p={p}: {speedup}-----------\n")


if __name__ == "__main__":
    main()