#include <iostream>  // Include the input-output stream library
#include <fstream>
using namespace std;

bool iv(string exp,string right,bool strike,bool bid,bool ask,bool underlying) {
	cout << "not working";
	return 0;
}

int main() {
	ofstream options;
	options.open ("./Data/options.csv" , ios::out);
	if (options.is_open()){
    		while ( getline (options, line) )
    		{
      			cout << line << '\n';
    		}
    		options.close();
  	}
cout << "Finished";  // Print a message to the screen
return 0;  // Return 0 to indicate the program ended successfully
}

