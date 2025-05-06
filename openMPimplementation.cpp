#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <string>
#include <unordered_set>

using namespace std;

const long long INF = numeric_limits<long long>::max();

struct Edge
{
    int targetVertex;
    long long weight;
    Edge() : targetVertex(0), weight(0) {}
    Edge(int target, long long w) : targetVertex(target), weight(w) {}
};

struct Compare
{
    bool operator()(const pair<long long, int> &a, const pair<long long, int> &b)
    {
        return a.first > b.first;
    }
};

void exportToCSV(const vector<long long> &distances, const string &filename, int numVertices)
{
    ofstream outfile(filename);
    if (!outfile.is_open())
    {
        cerr << "Error: Could not open " << filename << " for writing" << endl;
        return;
    }
    outfile << "Vertex,Distance\n";
#pragma omp parallel for
    for (int i = 1; i <= numVertices; ++i)
    {
        stringstream ss;
        ss << i << ",";
        if (distances[i] == INF)
            ss << "INF";
        else
            ss << distances[i];
        ss << "\n";
#pragma omp critical
        outfile << ss.str();
    }
    outfile.close();
}

vector<vector<Edge>> readGraph(const string &filename, int &numVertices)
{
    unordered_set<int> vertexSet;
    vector<tuple<int, int, long long>> edges;

    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error: Could not open file " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    string line;
    while (getline(file, line))
    {
        istringstream iss(line);
        int src, dst;
        long long weight;
        if (!(iss >> src >> dst >> weight))
        {
            cerr << "Error: Invalid line in file: " << line << endl;
            continue;
        }
        if (src <= 0 || dst <= 0)
        {
            cerr << "Error: Invalid vertex ID in file: " << src << " or " << dst << endl;
            continue;
        }
        if (weight < 0)
        {
            cerr << "Error: Negative weights not supported: " << weight << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
#pragma omp critical
        {
            vertexSet.insert(src);
            vertexSet.insert(dst);
            edges.emplace_back(src, dst, weight);
        }
    }
    file.close();

    numVertices = vertexSet.empty() ? 0 : *max_element(vertexSet.begin(), vertexSet.end());
    vector<vector<Edge>> graph(numVertices + 1);

#pragma omp parallel for
    for (size_t i = 0; i < edges.size(); ++i)
    {
        auto [src, dst, weight] = edges[i];
#pragma omp critical
        {
            graph[src].emplace_back(dst, weight);
            graph[dst].emplace_back(src, weight);
        }
    }

    return graph;
}

void distributeGraph(int rank, int size, const vector<vector<Edge>> &fullGraph, int numVertices,
                     vector<vector<Edge>> &localGraph, int &localStart, int &localCount)
{
    int nodesPerProc = numVertices / size;
    int remainder = numVertices % size;

    localStart = rank * nodesPerProc + 1;
    localCount = (rank < remainder) ? nodesPerProc + 1 : nodesPerProc;
    if (rank >= remainder)
    {
        localStart += remainder;
    }

    if (rank == size - 1)
    {
        localCount = numVertices - localStart + 1;
    }

    localGraph.resize(localCount + 1);

    if (rank == 0)
    {
#pragma omp parallel for
        for (int v = localStart; v < localStart + localCount; ++v)
        {
            if (v < fullGraph.size())
                localGraph[v - localStart + 1] = fullGraph[v];
        }

        for (int r = 1; r < size; ++r)
        {
            int procStart = r * nodesPerProc + 1;
            int procCount = (r < remainder) ? nodesPerProc + 1 : nodesPerProc;
            if (r >= remainder)
            {
                procStart += remainder;
            }
            if (r == size - 1)
            {
                procCount = numVertices - procStart + 1;
            }

            for (int v = procStart; v < procStart + procCount; ++v)
            {
                int edgeCount = (v < fullGraph.size()) ? fullGraph[v].size() : 0;
                MPI_Send(&edgeCount, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                if (edgeCount > 0)
                {
                    vector<int> targets(edgeCount);
                    vector<long long> weights(edgeCount);
#pragma omp parallel for
                    for (int j = 0; j < edgeCount; ++j)
                    {
                        targets[j] = fullGraph[v][j].targetVertex;
                        weights[j] = fullGraph[v][j].weight;
                    }
                    MPI_Send(targets.data(), edgeCount, MPI_INT, r, 1, MPI_COMM_WORLD);
                    MPI_Send(weights.data(), edgeCount, MPI_LONG_LONG, r, 2, MPI_COMM_WORLD);
                }
            }
        }
    }
    else
    {
        for (int v = localStart; v < localStart + localCount; ++v)
        {
            int edgeCount;
            MPI_Recv(&edgeCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            localGraph[v - localStart + 1].resize(edgeCount);
            if (edgeCount > 0)
            {
                vector<int> targets(edgeCount);
                vector<long long> weights(edgeCount);
                MPI_Recv(targets.data(), edgeCount, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(weights.data(), edgeCount, MPI_LONG_LONG, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#pragma omp parallel for
                for (int j = 0; j < edgeCount; ++j)
                {
                    localGraph[v - localStart + 1][j] = Edge(targets[j], weights[j]);
                }
            }
        }
    }
}

int getOwnerRank(int vertex, int numVertices, int size)
{
    int nodesPerProc = numVertices / size;
    int remainder = numVertices % size;
    int rank = 0;
    int currentVertex = 1;

    for (int r = 0; r < size; ++r)
    {
        int count = (r < remainder) ? nodesPerProc + 1 : nodesPerProc;
        if (r == size - 1)
        {
            count = numVertices - currentVertex + 1;
        }
        if (vertex >= currentVertex && vertex < currentVertex + count)
        {
            return r;
        }
        currentVertex += count;
    }
    return size - 1;
}

void parallelDijkstra(int sourceVertex, int rank, int size, int numVertices,
                      const vector<vector<Edge>> &localGraph, int localStart, int localCount,
                      vector<long long> &localDist, vector<int> &localParent)
{
    localDist.assign(localCount + 1, INF);
    localParent.assign(localCount + 1, -1);
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, Compare> localPQ;

    if (sourceVertex >= localStart && sourceVertex < localStart + localCount)
    {
        int localVertex = sourceVertex - localStart + 1;
        if (localVertex < localDist.size())
        {
            localDist[localVertex] = 0;
            localParent[localVertex] = sourceVertex;
#pragma omp critical
            localPQ.push({0, sourceVertex});
        }
    }

    vector<bool> visited(numVertices + 1, false);
    bool done = false;

    while (!done)
    {
        long long localMinDist = INF;
        int localMinNode = -1;
#pragma omp critical
        if (!localPQ.empty())
        {
            localMinDist = localPQ.top().first;
            localMinNode = localPQ.top().second;
            localPQ.pop();
        }

        long long globalMinDist;
        MPI_Allreduce(&localMinDist, &globalMinDist, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);

        if (globalMinDist == INF)
        {
            done = true;
            continue;
        }

        int globalMinNode = -1;
        int ownerRank = -1;
        if (localMinDist == globalMinDist && localMinNode != -1)
        {
            globalMinNode = localMinNode;
            ownerRank = rank;
        }

        struct NodeRank
        {
            int node;
            int rank;
        } localNodeRank = {globalMinNode, ownerRank}, globalNodeRank;

        MPI_Allreduce(&localNodeRank, &globalNodeRank, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        globalMinNode = globalNodeRank.node;
        ownerRank = globalNodeRank.rank;

        if (globalMinNode == -1 || visited[globalMinNode])
        {
            continue;
        }

#pragma omp critical
        visited[globalMinNode] = true;

        if (rank == ownerRank)
        {
            int localVertex = globalMinNode - localStart + 1;
            if (localVertex >= 1 && localVertex <= localCount && localVertex < localGraph.size())
            {
#pragma omp parallel
                {
                    vector<pair<long long, int>> localUpdates;
#pragma omp for
                    for (size_t i = 0; i < localGraph[localVertex].size(); ++i)
                    {
                        const Edge &edge = localGraph[localVertex][i];
                        int neighbor = edge.targetVertex;
                        long long altDist = globalMinDist + edge.weight;

                        int neighborOwner = getOwnerRank(neighbor, numVertices, size);
                        if (neighborOwner == rank)
                        {
                            int localNeighbor = neighbor - localStart + 1;
                            if (localNeighbor >= 1 && localNeighbor < localDist.size() && altDist < localDist[localNeighbor])
                            {
#pragma omp critical
                                {
                                    localDist[localNeighbor] = altDist;
                                    localParent[localNeighbor] = globalMinNode;
                                    localUpdates.emplace_back(altDist, neighbor);
                                }
                            }
                        }
                        else
                        {
                            MPI_Send(&altDist, 1, MPI_LONG_LONG, neighborOwner, 0, MPI_COMM_WORLD);
                            MPI_Send(&neighbor, 1, MPI_INT, neighborOwner, 1, MPI_COMM_WORLD);
                            MPI_Send(&globalMinNode, 1, MPI_INT, neighborOwner, 2, MPI_COMM_WORLD);
                        }
                    }
#pragma omp critical
                    for (const auto &update : localUpdates)
                    {
                        localPQ.push(update);
                    }
                }
            }
        }

        MPI_Status status;
        int flag;
        while (true)
        {
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
            if (!flag)
                break;

            if (status.MPI_TAG == 0)
            {
                long long altDist;
                MPI_Recv(&altDist, 1, MPI_LONG_LONG, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int neighbor;
                MPI_Recv(&neighbor, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int newParent;
                MPI_Recv(&newParent, 1, MPI_INT, status.MPI_SOURCE, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int neighborOwner = getOwnerRank(neighbor, numVertices, size);
                if (neighborOwner == rank)
                {
                    int localNeighbor = neighbor - localStart + 1;
                    if (localNeighbor >= 1 && localNeighbor < localDist.size() && altDist < localDist[localNeighbor])
                    {
#pragma omp critical
                        {
                            localDist[localNeighbor] = altDist;
                            localParent[localNeighbor] = newParent;
                            localPQ.push({altDist, neighbor});
                        }
                    }
                }
            }
        }
    }
}

void deleteEdgeAndUpdate(vector<vector<Edge>> &graph, vector<long long> &dist, vector<int> &parent,
                         int u, int v, int rank, int size, int localStart, int localCount, int numVertices)
{
    // Remove edge u->v if it exists in this process's portion
    if (u >= localStart && u < localStart + localCount)
    {
        int localU = u - localStart + 1;
        if (localU < graph.size())
        {
            auto &edges = graph[localU];
#pragma omp critical
            edges.erase(remove_if(edges.begin(), edges.end(),
                                  [v](const Edge &e)
                                  { return e.targetVertex == v; }),
                        edges.end());
        }
    }

    // Remove edge v->u if it exists in this process's portion
    if (v >= localStart && v < localStart + localCount)
    {
        int localV = v - localStart + 1;
        if (localV < graph.size())
        {
            auto &edges = graph[localV];
#pragma omp critical
            edges.erase(remove_if(edges.begin(), edges.end(),
                                  [u](const Edge &e)
                                  { return e.targetVertex == u; }),
                        edges.end());
        }
    }

    // Check if the deleted edge was part of any shortest path
    bool needUpdate = false;
    if (u >= localStart && u < localStart + localCount)
    {
        int localU = u - localStart + 1;
        if (localU < parent.size() && parent[localU] == v)
            needUpdate = true;
    }
    if (v >= localStart && v < localStart + localCount)
    {
        int localV = v - localStart + 1;
        if (localV < parent.size() && parent[localV] == u)
            needUpdate = true;
    }

    // If the edge was part of any shortest path, recompute affected distances
    if (needUpdate)
    {
        priority_queue<pair<long long, int>, vector<pair<long long, int>>, Compare> pq;

        // Add affected vertices to the queue
        if (u >= localStart && u < localStart + localCount)
        {
            int localU = u - localStart + 1;
            if (localU < dist.size())
            {
#pragma omp critical
                pq.push({dist[localU], u});
            }
        }
        if (v >= localStart && v < localStart + localCount)
        {
            int localV = v - localStart + 1;
            if (localV < dist.size())
            {
#pragma omp critical
                pq.push({dist[localV], v});
            }
        }

        // Run Dijkstra's update for affected vertices
        while (!pq.empty())
        {
            long long currentDist;
            int currentVertex;
#pragma omp critical
            {
                if (!pq.empty())
                {
                    currentDist = pq.top().first;
                    currentVertex = pq.top().second;
                    pq.pop();
                }
                else
                {
                    currentDist = INF;
                    currentVertex = -1;
                }
            }

            if (currentVertex == -1)
                break;

            if (currentVertex >= localStart && currentVertex < localStart + localCount)
            {
                int localVertex = currentVertex - localStart + 1;
                if (localVertex >= 1 && localVertex < graph.size())
                {
#pragma omp parallel
                    {
                        vector<pair<long long, int>> localUpdates;
#pragma omp for
                        for (size_t i = 0; i < graph[localVertex].size(); ++i)
                        {
                            const Edge &edge = graph[localVertex][i];
                            int neighbor = edge.targetVertex;
                            long long newDist = currentDist + edge.weight;

                            int neighborOwner = getOwnerRank(neighbor, numVertices, size);
                            if (neighborOwner == rank)
                            {
                                int localNeighbor = neighbor - localStart + 1;
                                if (localNeighbor >= 1 && localNeighbor < dist.size() && newDist < dist[localNeighbor])
                                {
#pragma omp critical
                                    {
                                        dist[localNeighbor] = newDist;
                                        parent[localNeighbor] = currentVertex;
                                        localUpdates.emplace_back(newDist, neighbor);
                                    }
                                }
                            }
                        }
#pragma omp critical
                        for (const auto &update : localUpdates)
                        {
                            pq.push(update);
                        }
                    }
                }
            }
        }
    }
}

void newInsertEdgeAndUpdate(vector<vector<Edge>> &graph, vector<long long> &dist, vector<int> &parent,
                            int u, int v, long long weight, int rank, int size, int localStart, int localCount, int numVertices)
{
    // Add edge u->v if u is in this process's portion
    if (u >= localStart && u < localStart + localCount)
    {
        int localU = u - localStart + 1;
        if (localU < graph.size())
        {
#pragma omp critical
            graph[localU].emplace_back(v, weight);
        }
    }

    // Add edge v->u if v is in this process's portion (undirected graph)
    if (v >= localStart && v < localStart + localCount)
    {
        int localV = v - localStart + 1;
        if (localV < graph.size())
        {
#pragma omp critical
            graph[localV].emplace_back(u, weight);
        }
    }

    // Initialize priority queue for updating shortest paths
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, Compare> pq;

    // Add vertices u and v to the queue if they are in this process's portion
    if (u >= localStart && u < localStart + localCount)
    {
        int localU = u - localStart + 1;
        if (localU < dist.size())
        {
#pragma omp critical
            pq.push({dist[localU], u});
        }
    }
    if (v >= localStart && v < localStart + localCount)
    {
        int localV = v - localStart + 1;
        if (localV < dist.size())
        {
#pragma omp critical
            pq.push({dist[localV], v});
        }
    }

    // Update shortest paths considering the new edge
    while (!pq.empty())
    {
        long long currentDist;
        int currentVertex;
#pragma omp critical
        {
            if (!pq.empty())
            {
                currentDist = pq.top().first;
                currentVertex = pq.top().second;
                pq.pop();
            }
            else
            {
                currentDist = INF;
                currentVertex = -1;
            }
        }

        if (currentVertex == -1)
            break;

        if (currentVertex >= localStart && currentVertex < localStart + localCount)
        {
            int localVertex = currentVertex - localStart + 1;
            if (localVertex >= 1 && localVertex < graph.size())
            {
#pragma omp parallel
                {
                    vector<pair<long long, int>> localUpdates;
#pragma omp for
                    for (size_t i = 0; i < graph[localVertex].size(); ++i)
                    {
                        const Edge &edge = graph[localVertex][i];
                        int neighbor = edge.targetVertex;
                        long long newDist = currentDist + edge.weight;

                        int neighborOwner = getOwnerRank(neighbor, numVertices, size);
                        if (neighborOwner == rank)
                        {
                            int localNeighbor = neighbor - localStart + 1;
                            if (localNeighbor >= 1 && localNeighbor < dist.size() && newDist < dist[localNeighbor])
                            {
#pragma omp critical
                                {
                                    dist[localNeighbor] = newDist;
                                    parent[localNeighbor] = currentVertex;
                                    localUpdates.emplace_back(newDist, neighbor);
                                }
                            }
                        }
                    }
#pragma omp critical
                    for (const auto &update : localUpdates)
                    {
                        pq.push(update);
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE)
    {
        cerr << "MPI implementation does not support MPI_THREAD_MULTIPLE" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numVertices = 0;
    vector<vector<Edge>> fullGraph, localGraph;
    vector<long long> localDist;
    vector<int> localParent;
    int localStart, localCount;

    if (rank == 0)
    {
        string filename = "./dense3.txt";
        if (argc > 1)
        {
            filename = argv[1];
        }
        fullGraph = readGraph(filename, numVertices);
        cout << "Graph loaded with " << numVertices << " vertices." << endl;
    }

    MPI_Bcast(&numVertices, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (numVertices < size)
    {
        if (rank == 0)
        {
            cerr << "Error: Too many processes (" << size << ") for " << numVertices
                 << " vertices. Use fewer processes." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0)
    {
        distributeGraph(rank, size, fullGraph, numVertices, localGraph, localStart, localCount);
    }
    else
    {
        fullGraph.clear();
        distributeGraph(rank, size, fullGraph, numVertices, localGraph, localStart, localCount);
    }

    int sourceVertex = 1;
    if (argc > 2)
    {
        sourceVertex = atoi(argv[2]);
    }
    if (sourceVertex < 1 || sourceVertex > numVertices)
    {
        if (rank == 0)
        {
            cerr << "Error: Invalid source vertex " << sourceVertex
                 << ". Must be between 1 and " << numVertices << "." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0)
    {
        cout << "Starting Dijkstra's algorithm from source vertex " << sourceVertex << endl;
    }

    auto startTime = chrono::high_resolution_clock::now();
    parallelDijkstra(sourceVertex, rank, size, numVertices, localGraph, localStart, localCount, localDist, localParent);
    auto endTime = chrono::high_resolution_clock::now();

    if (rank == 0)
    {
        chrono::duration<double> elapsed = endTime - startTime;
        cout << "SSSP computation time: " << elapsed.count() << " seconds" << endl;
    }

    vector<long long> globalDist(numVertices + 1, INF);
    vector<int> globalParent(numVertices + 1, -1);
    vector<int> recvCounts(size), displs(size);

#pragma omp parallel for
    for (int r = 0; r < size; ++r)
    {
        int nodesPerProc = numVertices / size;
        int remainder = numVertices % size;
        int procStart = r * nodesPerProc + 1;
        int procCount = (r < remainder) ? nodesPerProc + 1 : nodesPerProc;
        if (r >= remainder)
        {
            procStart += remainder;
        }
        if (r == size - 1)
        {
            procCount = numVertices - procStart + 1;
        }
        recvCounts[r] = procCount;
        displs[r] = procStart - 1;
    }

    if (localCount > 0)
    {
        MPI_Gatherv(localDist.data() + 1, localCount, MPI_LONG_LONG,
                    globalDist.data() + 1, recvCounts.data(), displs.data(), MPI_LONG_LONG,
                    0, MPI_COMM_WORLD);
        MPI_Gatherv(localParent.data() + 1, localCount, MPI_INT,
                    globalParent.data() + 1, recvCounts.data(), displs.data(), MPI_INT,
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
        globalDist[sourceVertex] = 0;
        globalParent[sourceVertex] = sourceVertex;
        exportToCSV(globalDist, "initial_distances_openMP.csv", numVertices);

        // Perform edge deletion
        auto start_deletion = chrono::high_resolution_clock::now();
        deleteEdgeAndUpdate(localGraph, globalDist, globalParent, 16395, 11, rank, size, localStart, localCount, numVertices);
        auto end_deletion = chrono::high_resolution_clock::now();
        cout << "Deletion update time: " << chrono::duration<double>(end_deletion - start_deletion).count() << " seconds" << endl;
        exportToCSV(globalDist, "after_deletion_OpenMP.csv", numVertices);

        // Perform edge insertion
        auto start_insertion = chrono::high_resolution_clock::now();
        newInsertEdgeAndUpdate(localGraph, globalDist, globalParent, 16398, 12, 67000, rank, size, localStart, localCount, numVertices);
        auto end_insertion = chrono::high_resolution_clock::now();
        cout << "Insertion update time: " << chrono::duration<double>(end_insertion - start_insertion).count() << " seconds" << endl;
        exportToCSV(globalDist, "openMpafterinsertion.csv", numVertices);
    }

    MPI_Finalize();
    return 0;
}