from mpi4py import MPI
import numpy as np
import time

#  This method generates random data for matrices A and B and distributes it among the different processes. 
def distribute_data(n, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Calculate the number of rows each process will receive
    local_n = n // size
    # Calulate the number of leftover rows
    remainder = n % size
    # Add one row to each process until there are no more leftover rows
    local_n += 1 if rank < remainder else 0

    A = np.random.rand(local_n, n)
    B = np.random.rand(n, local_n)

    return A, B

# This method performs matrix multiplication. Pretty simple.
def matrix_multiply(A, B):
    return np.dot(A, B)

def main():
    # Sets up the MPI communicator object which represents all processes in the MPI world
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Values of n (matrix size) and p (number of processes)
    n_values = [100, 1000, 5000, 10000]
    p_values = [1, 2, 4, 8]

    for n in n_values:
        for p in p_values:
            # Start the timer on the root process
            if rank == 0:
                start_time = time.time()

            # Distribute the data among the processes
            A, B = distribute_data(n, comm)

            # Perform local matrix multiplication
            local_C = matrix_multiply(A, B)

            # Sum the partial results to obtain the final result
            comm.Allreduce(MPI.IN_PLACE, local_C, op=MPI.SUM)

            # End the timer on the root process
            if rank == 0:
                end_time = time.time()
                runtime = end_time - start_time
                print(f"\nRuntime for n={n}, p={p}: {runtime} seconds\n")
                
                # If p = 1, store the runtime as the sequential time as there is no parallel time to compare to
                if p == 1:
                    sequential_time = runtime
                    
                    # Calculate the speedup for every other process
                else:
                    parallel_time = runtime
                    speedup = sequential_time / parallel_time
                    print(f"\n-----------Speedup for n={n}, p={p}: {speedup}-----------\n")


if __name__ == "__main__":
    main()