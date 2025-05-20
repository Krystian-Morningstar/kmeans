#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <cstring>
#include "mpi.h"
using namespace std;

string kmeans(int k, int rank, double data[][3], int numPoints)
{
    int maxAttrib = 2;
    int maxC = k;

    double myData[numPoints][maxAttrib + 2];
    double myDataTemp[numPoints][maxAttrib + 2];
    double myCs[maxC][maxAttrib + 1];
    double myClog[maxC][maxAttrib + 1];

    for (int i = 0; i < numPoints; i++) {
        myData[i][0] = data[i][0];
        myData[i][1] = data[i][1];
        myData[i][2] = data[i][2];
    }

    for (int i = 0; i < numPoints; i++) {
        myDataTemp[i][0] = i + 1;
        for (int j = 1; j <= maxAttrib + 1; j++) {
            myDataTemp[i][j] = 0;
        }
    }

    for (int i = 0; i < maxC; i++) {
        for (int j = 0; j < maxAttrib + 1; j++) {
            myClog[i][j] = 0;
            myCs[i][j] = myData[i % numPoints][j];
        }
    }

    bool isEnd = true;
    int it = 0;

    do {
        for (int i = 0; i < numPoints; i++) {
            for (int j = 1; j <= maxC; j++) {
                double sumValue = 0;
                for (int t = 1; t <= maxAttrib; t++) {
                    sumValue += pow(myCs[j - 1][t] - myData[i][t], 2);
                }
                myDataTemp[i][j] = sqrt(sumValue);
            }
        }

        for (int i = 0; i < numPoints; i++) {
            double minVal = myDataTemp[i][1];
            int minPos = 1;
            for (int j = 2; j <= maxC; j++) {
                if (myDataTemp[i][j] < minVal) {
                    minVal = myDataTemp[i][j];
                    minPos = j;
                }
            }
            myData[i][maxAttrib + 1] = minPos;
        }

        for (int cent = 1; cent <= maxC; cent++) {
            for (int atrib = 1; atrib <= maxAttrib; atrib++) {
                double sum = 0;
                int count = 0;
                for (int i = 0; i < numPoints; i++) {
                    if ((int)myData[i][maxAttrib + 1] == cent) {
                        sum += myData[i][atrib];
                        count++;
                    }
                }
                if (count > 0)
                    myCs[cent - 1][atrib] = sum / count;
            }
        }

        isEnd = true;
        for (int cent = 0; cent < maxC; cent++) {
            for (int atrib = 1; atrib <= maxAttrib; atrib++) {
                if (myClog[cent][atrib] != myCs[cent][atrib]) {
                    isEnd = false;
                }
                myClog[cent][atrib] = myCs[cent][atrib];
            }
        }

        it++;
    } while ((it < 10) && (!isEnd));

    stringstream result;
    result << "\n>>> Resultados del nodo " << rank << " (k=" << k << "):\n";
    for (int clu = 0; clu < maxC; clu++) {
        result << "Cluster " << clu + 1 << ": ";
        for (int i = 0; i < numPoints; i++) {
            if ((int)myData[i][maxAttrib + 1] == clu + 1) {
                result << myData[i][0] << " ";
            }
        }
        result << "\n";
    }

    return result.str();
}

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int maxM = 9;
    const int maxAttrib = 2;
    const int cols = maxAttrib + 1;

    double fullData[maxM][cols] = {
        {1.0, 0.8, 1.8},
        {2.0, 1.1, 1.3},
        {3.0, 0.8, 0.9},
        {4.0, 1.0, 0.8},
        {5.0, 1.4, 1.2},
        {6.0, 1.5, 1.3},
        {7.0, 1.1, 0.7},
        {8.0, 0.5, 1.8},
        {9.0, 1.5, 1.2}
    };

    double localData[5][cols];
    int localCount;

    if (rank == 0) {
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < cols; j++)
                localData[i][j] = fullData[i][j];
        localCount = 5;

        MPI_Send(fullData[5], 4 * cols, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Recv(localData, 4 * cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        localCount = 4;
    }

    int k = rank + 2;
    string localResult = kmeans(k, rank, localData, localCount);

    if (rank == 1) {
        int len = localResult.size();
        MPI_Send(&len, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(localResult.c_str(), len, MPI_CHAR, 0, 2, MPI_COMM_WORLD);
    } else if (rank == 0) {
        cout << localResult << flush;

        int recvLen;
        MPI_Recv(&recvLen, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        char *recvBuf = new char[recvLen + 1];
        MPI_Recv(recvBuf, recvLen, MPI_CHAR, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        recvBuf[recvLen] = '\0';

        cout << recvBuf << flush;
        delete[] recvBuf;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
