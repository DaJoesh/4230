from mpi4py import MPI
import numpy as np
import time

# This method generates random data for matrices A and B and distributes it among the different processors
def distribute_data(n, rank, num_processors):
    
    # Calculate the number of rows each processor will handle
    local_n = n // num_processors
    
    # Calculate the number of leftover rows
    remainder = n % num_processors
    
    # Add one row to each processor until there are no more leftover rows
    local_n += 1 if rank < remainder else 0

    A = np.random.rand(local_n, n)
    B = np.random.rand(n, local_n)

    return A, B

# This method performs matrix multiplication for the assigned portion of matrices A and B
def matrix_multiply(A, B):
    return np.dot(A, B)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_processors = comm.Get_size()

    print(f"comm size: {num_processors}")

    n_values = [100, 1000, 5000, 10000]
    num_processors_values = [1, 2, 4, 8]

    for n in n_values:
        for num_procs in num_processors_values:
            # Skips this iteration if the number of processors is less than the specified number
            if num_procs > num_processors:
                continue 
            
            # Only executes the loop for the assigned number of processors
            if rank < num_procs:
                start_time = time.time()

                A, B = distribute_data(n, rank, num_procs)

                local_C = matrix_multiply(A, B)

                # Gather all local_C matrices on rank 0
                all_local_C = comm.gather(local_C, root=0)

                if rank == 0:
                    # Concatenate all_local_C matrices to form the global_C matrix
                    global_C = np.concatenate(all_local_C, axis=1)
                    
                    # Print to text file
                    with open(f"output_{n}_{num_procs}.txt", "w") as f:
                        f.write(f"Matrix A (shape {n}x{n}):\n{A}\n\n")
                        f.write(f"Matrix B (shape {n}x{n}):\n{B}\n\n")
                        f.write(f"Matrix C (shape {n}x{n}):\n{global_C}\n")
                    
                    # Prints the runtime
                    end_time = time.time()
                    runtime = end_time - start_time
                    print(f"\nRuntime for n={n}, num_procs={num_procs}: {runtime} seconds\n")

                    # Calculates the speedup (sequential_time/parallel_time) only if the number of processors is greater than 1
                    if num_procs == 1:
                        sequential_time = runtime
                    else:
                        parallel_time = runtime
                        speedup = sequential_time / parallel_time
                        print(f"\n-----------Speedup for n={n}, num_procs={num_procs}: {speedup}-----------\n")

if __name__ == "__main__":
    main()