# Parallel-Algorithms-for-Updating-Single-Source-Shortest-Paths-in-Large-Scale-Dynamic-Networks

Dijkstra's Algorithm Implementations

This repository contains three implementations of Dijkstra's Single Source Shortest Path (SSSP) algorithm for undirected weighted graphs:





Serial: A single-threaded implementation.



MPI: A distributed-memory parallel implementation using the Message Passing Interface (MPI).



OpenMP + MPI: A hybrid implementation combining shared-memory parallelism (OpenMP) with distributed-memory parallelism (MPI).

The implementations support initial SSSP computation, edge deletion updates, and edge insertion updates, tested on a graph with 19,999 vertices.

Prerequisites

To compile and run the code, ensure the following dependencies are installed:





C++ Compiler: GCC or any C++11-compliant compiler.



MPI Library: OpenMPI or MPICH for the MPI and OpenMP+MPI implementations.



OpenMP: Supported by the compiler (e.g., GCC with -fopenmp flag) for the OpenMP+MPI implementation.



Standard Libraries: <iostream>, <vector>, <queue>, <limits>, <fstream>, <sstream>, <chrono>, <algorithm>, <string>, <unordered_set>.

Repository Structure

├── serialImplementation.cpp    # Serial implementation of Dijkstra's algorithm
├── MPIimplementation.cpp       # MPI-based parallel implementation
├── openMPimplementation.cpp    # Hybrid OpenMP+MPI implementation
├── dense3.txt                  # Sample input graph file
└── README.md                   # This file

Input Format

The input graph is provided in a text file (e.g., dense3.txt) with the following format:

u v w





u: Source vertex ID (integer).



v: Target vertex ID (integer).



w: Edge weight (long long integer).

Each line represents an undirected edge between vertices u and v with weight w.

Compilation and Execution

Serial Implementation

g++ -o serial serialImplementation.cpp
./serial





Reads the graph from dense3.txt.



Outputs execution times and CSV files (initial_distances.csv, after_deletion.csv, after_insertion.csv) with vertex distances.

MPI Implementation

mpicxx -o mpi MPIimplementation.cpp
mpirun -np <num_processes> ./mpi [input_file] [source_vertex]





Default input: dense3.txt, source vertex: 1.



Example: mpirun -np 4 ./mpi dense3.txt 1.



Outputs execution times and CSV files (initial_distances_MPI.csv, after_deletion_mpi.csv, Mpiafterinsertion.csv).



Ensure <num_processes> does not exceed the number of vertices.

OpenMP + MPI Implementation

mpicxx -o openmp_mpi openMPimplementation.cpp -fopenmp
mpirun -np <num_processes> ./openmp_mpi [input_file] [source_vertex]





Default input: dense3.txt, source vertex: 1.



Example: mpirun -np 4 ./openmp_mpi dense3.txt 1.



Outputs execution times and CSV files (initial_distances_openMP.csv, after_deletion_OpenMP.csv, openMpafterinsertion.csv).



Requires MPI with thread support (MPI_THREAD_MULTIPLE).

Output

Each implementation generates CSV files with the following format:

Vertex,Distance
1,0
2,INF
...





Vertex: Vertex ID.



Distance: Shortest path distance from the source vertex (INF if unreachable).

Console output includes execution times for:





Initial SSSP computation.



Edge deletion update (removing edge between vertices 16395 and 11).



Edge insertion update (adding edge between vertices 16398 and 12 with weight 67000).

Performance

The implementations were tested on a graph with 19,999 vertices. Key performance metrics:





Serial:





SSSP: 0.178279 s



Deletion: 0.188125 s



Insertion: 2.838e-06 s



MPI:





SSSP: 0.0410053 s



Deletion: 1.74e-06 s



Insertion: 2.507e-06 s



OpenMP + MPI:





SSSP: 0.305092 s



Deletion: 8.601e-06 s



Insertion: 5.8907e-05 s

The MPI implementation is the fastest due to efficient parallelization, while the serial implementation is slowest due to its sequential nature. The OpenMP+MPI implementation underperforms due to thread management overhead.

Notes





The graph must have positive edge weights, as Dijkstra's algorithm does not support negative weights.



The MPI and OpenMP+MPI implementations require the number of processes to be less than or equal to the number of vertices.



The OpenMP+MPI implementation requires an MPI library supporting MPI_THREAD_MULTIPLE.

