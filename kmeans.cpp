#include <iostream>
#include <sstream>
#include <cmath>
#include "mpi.h"
using namespace std;

void kmeans(int k, int rank)
{
    int maxM = 9;
    int maxAttrib = 2;
    int maxC = k;

    double myData[maxM][maxAttrib + 2];
    double myDataTemp[maxM][maxAttrib + 2];
    double myCs[maxC][maxAttrib + 1];
    double myClog[maxC][maxAttrib + 1];

    // Fill data
    myData[0][0] = 1.0; myData[0][1] = 0.8; myData[0][2] = 1.8;
    myData[1][0] = 2.0; myData[1][1] = 1.1; myData[1][2] = 1.3;
    myData[2][0] = 3.0; myData[2][1] = 0.8; myData[2][2] = 0.9;
    myData[3][0] = 4.0; myData[3][1] = 1.0; myData[3][2] = 0.8;
    myData[4][0] = 5.0; myData[4][1] = 1.4; myData[4][2] = 1.2;
    myData[5][0] = 6.0; myData[5][1] = 1.5; myData[5][2] = 1.3;
    myData[6][0] = 7.0; myData[6][1] = 1.1; myData[6][2] = 0.7;
    myData[7][0] = 8.0; myData[7][1] = 0.5; myData[7][2] = 1.8;
    myData[8][0] = 9.0; myData[8][1] = 1.5; myData[8][2] = 1.2;

    for (int i = 0; i < maxM; i++) {
        myDataTemp[i][0] = i + 1;
        for (int j = 1; j <= maxAttrib + 1; j++) {
            myDataTemp[i][j] = 0;
        }
    }

    for (int i = 0; i < maxC; i++) {
        for (int j = 0; j < maxAttrib + 1; j++) {
            myClog[i][j] = 0;
            myCs[i][j] = myData[i][j];
        }
    }

    bool isEnd = true;
    int it = 0;

    do {
        for (int i = 0; i < maxM; i++) {
            for (int j = 1; j <= maxC; j++) {
                double sumValue = 0;
                for (int t = 1; t <= maxAttrib; t++) {
                    sumValue += pow(myCs[j - 1][t] - myData[i][t], 2);
                }
                myDataTemp[i][j] = sqrt(sumValue);
            }
        }

        for (int i = 0; i < maxM; i++) {
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
                for (int i = 0; i < maxM; i++) {
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
        for (int i = 0; i < maxM; i++) {
            if ((int)myData[i][maxAttrib + 1] == clu + 1) {
                result << myData[i][0] << " ";
            }
        }
        result << "\n";
    }

    cout << result.str() << flush;
}

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "Nodo " << rank << " iniciado..." << endl << flush;

    int k = rank + 2; // Por ejemplo: nodo 0 usa k=2, nodo 1 usa k=3
    kmeans(k, rank);

    MPI_Barrier(MPI_COMM_WORLD); // Espera a que todos terminen
    cout << "Nodo " << rank << " finalizado." << endl << flush;

    MPI_Finalize();
    return 0;
}

