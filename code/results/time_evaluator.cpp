#include <string>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <array>
#include <fstream>

using namespace std;

int main()
{
    string command("cd Parallel-Histogram-Equalization/OMP_histogram_equalization/cmake-build-debug && ./OMP_histogram_equalization");

    std::array<char, 128> buffer;

    int n = 10;

    char* result[n];       //array of char* that contains the execution times

    ofstream outfile;
    outfile.open("path_to_file/par100x100.csv");

    double sum = 0;
    int count = 0;

    for (int i = 0; i < n; i++){

        FILE* pipe = popen(command.c_str(), "r");
        if (!pipe){

            std::cerr << "Couldn't start command." << std::endl;
            return 0;
        }

       
        while (fgets(buffer.data(), 128, pipe) != NULL) {
            result[i] = buffer.data();
            cout << result[i];

            float num = atof(result[i]);

            sum += num;
        }

    }

    outfile << sum / n << endl;



    cout << "created file" <<endl;
    outfile.close();


}