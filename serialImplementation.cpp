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
    int to;
    long long weight;
    Edge(int t, long long w) : to(t), weight(w) {}
};

void exportToCSV(const vector<long long> &dist, const string &filename)
{
    ofstream outfile(filename);
    outfile << "Vertex,Distance\n";
    for (size_t i = 1; i < dist.size(); ++i)
    {
        outfile << i << ",";
        outfile << (dist[i] == INF ? "INF" : to_string(dist[i])) << "\n";
    }
    outfile.close();
}

vector<vector<Edge>> readGraph(const string &filename, int &numVertices)
{
    unordered_set<int> vertexSet;
    vector<tuple<int, int, int>> edges;

    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error opening file " << filename << endl;
        exit(1);
    }

    int u, v, w;
    while (file >> u >> v >> w)
    {
        vertexSet.insert(u);
        vertexSet.insert(v);
        edges.emplace_back(u, v, w);
    }
    file.close();

    numVertices = *max_element(vertexSet.begin(), vertexSet.end());
    vector<vector<Edge>> graph(numVertices + 1);

    for (const auto &[u, v, wgt] : edges)
    {
        graph[u].emplace_back(v, wgt);
        graph[v].emplace_back(u, wgt);
    }

    return graph;
}

void dijkstra(const vector<vector<Edge>> &graph, int source, vector<long long> &dist, vector<int> &parent)
{
    int n = graph.size() - 1;
    dist.assign(n + 1, INF);
    parent.assign(n + 1, -1);
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;

    dist[source] = 0;
    pq.emplace(0, source);

    while (!pq.empty())
    {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u])
            continue;
        for (const auto &edge : graph[u])
        {
            if (dist[u] + edge.weight < dist[edge.to])
            {
                dist[edge.to] = dist[u] + edge.weight;
                parent[edge.to] = u;
                pq.emplace(dist[edge.to], edge.to);
            }
        }
    }
}

void deleteEdgeAndUpdate(vector<vector<Edge>> &graph, vector<long long> &dist, vector<int> &parent, int u, int v)
{
    auto removeEdge = [](vector<Edge> &edges, int target)
    {
        edges.erase(remove_if(edges.begin(), edges.end(),
                              [target](const Edge &e)
                              { return e.to == target; }),
                    edges.end());
    };
    removeEdge(graph[u], v);
    removeEdge(graph[v], u);

    if (parent[v] == u || parent[u] == v)
    {
        int numVertices = graph.size() - 1;

        queue<int> q;
        dist[v] = INF;
        parent[v] = -1;
        q.push(v);

        while (!q.empty())
        {
            int current = q.front();
            q.pop();
            for (const auto &edge : graph[current])
            {
                if (parent[edge.to] == current && dist[edge.to] != INF)
                {
                    dist[edge.to] = INF;
                    parent[edge.to] = -1;
                    q.push(edge.to);
                }
            }
        }

        priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
        for (int i = 1; i <= numVertices; ++i)
        {
            if (dist[i] != INF)
                pq.emplace(dist[i], i);
        }

        while (!pq.empty())
        {
            auto [d, u] = pq.top();
            pq.pop();
            if (d > dist[u])
                continue;
            for (const auto &edge : graph[u])
            {
                if (dist[u] + edge.weight < dist[edge.to])
                {
                    dist[edge.to] = dist[u] + edge.weight;
                    parent[edge.to] = u;
                    pq.emplace(dist[edge.to], edge.to);
                }
            }
        }
    }
}

void insertEdgeAndUpdate(vector<vector<Edge>> &graph, vector<long long> &dist, vector<int> &parent, int u, int v, long long w)
{
    graph[u].emplace_back(v, w);
    graph[v].emplace_back(u, w);

    bool updated = false;
    if (dist[u] != INF && dist[u] + w < dist[v])
    {
        dist[v] = dist[u] + w;
        parent[v] = u;
        updated = true;
    }
    if (dist[v] != INF && dist[v] + w < dist[u])
    {
        dist[u] = dist[v] + w;
        parent[u] = v;
        updated = true;
    }

    if (updated)
    {
        int numVertices = graph.size() - 1;
        priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
        for (int i = 1; i <= numVertices; ++i)
        {
            if (dist[i] != INF)
                pq.emplace(dist[i], i);
        }

        while (!pq.empty())
        {
            auto [d, u] = pq.top();
            pq.pop();
            if (d > dist[u])
                continue;
            for (const auto &edge : graph[u])
            {
                if (dist[u] + edge.weight < dist[edge.to])
                {
                    dist[edge.to] = dist[u] + edge.weight;
                    parent[edge.to] = u;
                    pq.emplace(dist[edge.to], edge.to);
                }
            }
        }
    }
}

int main()
{
    int numVertices;
    auto graph = readGraph("./dense3.txt", numVertices);
    vector<long long> dist;
    vector<int> parent;

    auto start_initial = chrono::high_resolution_clock::now();
    dijkstra(graph, 1, dist, parent);
    auto end_initial = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_initial = end_initial - start_initial;
    cout << "Initial SSSP computation time: " << elapsed_initial.count() << " seconds" << endl;
    exportToCSV(dist, "initial_distances.csv");

    // Uncomment and modify these to test
    auto start_deletion = chrono::high_resolution_clock::now();

    deleteEdgeAndUpdate(graph, dist, parent, 16395, 11);
    auto end_deletion = chrono::high_resolution_clock::now();
    cout << "Deletion update time: " << chrono::duration<double>(end_deletion - start_deletion).count() << " seconds" << endl;
    exportToCSV(dist, "after_deletion.csv");

    auto start_insertion = chrono::high_resolution_clock::now();
    insertEdgeAndUpdate(graph, dist, parent, 16398, 12, 67000);
    auto end_insertion = chrono::high_resolution_clock::now();
    cout << "Insertion update time: " << chrono::duration<double>(end_insertion - start_insertion).count() << " seconds" << endl;
    exportToCSV(dist, "after_insertion.csv");

    return 0;
}