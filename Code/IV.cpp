#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>  // For parsing the CSV line
using namespace std;

// Function to process option data (IV calculation or other processing)
int iv(string exp, string right, float strike, float bid, float ask, float underlying) {
    cout << "Exp: " << exp << ", Right: " << right << ", Strike: " << strike
         << ", Bid: " << bid << ", Ask: " << ask << ", Underlying: " << underlying << endl;
    // Example calculation or processing here; returning a placeholder value
    return 5;
}

int main() {
    string myText;
    ifstream MyReadFile("../Data/options.csv");

    if (!MyReadFile.is_open()) {
        cerr << "Failed to open file!" << endl;
        return 1;
    }

    // Read file line by line
    while (getline(MyReadFile, myText)) {
        stringstream ss(myText);  // Use a stringstream to parse the line
        string exp, right;
        float strike, bid, ask, underlying;

        if (getline(ss, exp, ',') && 
            getline(ss, right, ',') &&
            ss >> strike && ss.ignore(1) && 
            ss >> bid && ss.ignore(1) &&
            ss >> ask && ss.ignore(1) &&
            ss >> underlying) {

            iv(exp, right, strike, bid, ask, underlying);
        }

        sleep(0);
    }

    MyReadFile.close();
    return 0;
}

