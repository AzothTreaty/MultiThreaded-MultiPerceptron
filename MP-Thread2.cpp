//g++ -static -std=c++0x -pthread -fpermissive MP-Thread2.cpp -o MultiMP2.exe

#include <iostream>
#include <math.h>
#include <pthread.h>
#include <fstream>
#include <chrono>
#include <unistd.h>

using namespace std;
pthread_t* threadArray;
int* threadStates;// 0 == working, 1 == done
int threadLength, threadProcesses, threadInterval;//interval is numset/threadLength, threadProcesses is alittle more complex
double** globalNetwork, **globalInputs, **globalOutputs;

//'static' components of the MP
double bias, biasWeight, learningRate;
int numLayers, *numNodesInLayer;


void waitForThreads(){
	int returnVal;
	for(int q = 0; q < threadLength; q++){
		while(threadStates[q] != 1){
			//cout << "Waiting for thread #" << q << endl;
		}
	}
}

struct MP{
	double **network, **prevInputs, **deltas;
	double iterationErrorKo;

	void initOnce(){//static initialization of the MP class
		ifstream myStream;
		myStream.open("Config.txt", ifstream::in);
		bias = 1;
		biasWeight = 0.35;
		myStream >> numLayers;
		myStream >> learningRate;

		numNodesInLayer = new int[numLayers];
		globalNetwork = new double*[numLayers - 1];
		for(int q = 0; q < numLayers; q++){
			myStream >> numNodesInLayer[q];
		}
		
		for(int q = 0; q < numLayers - 1; q++){
			int numNodesNow = numNodesInLayer[q] * numNodesInLayer[q + 1];
			globalNetwork[q] = new double[numNodesNow];
			for(int w = 0; w < numNodesNow; w++){
				myStream >> globalNetwork[q][w];
			}
		}

		myStream.close();
	}
	void restart(){//gives it a fresh network to work with
		for(int q = 0; q < numLayers - 1; q++){
			int numNodesNow = numNodesInLayer[q] * numNodesInLayer[q + 1];
			network[q] = new double[numNodesNow];
			for(int w = 0; w < numNodesNow; w++){
				network[q][w] = globalNetwork[q][w];
			}
		}
		iterationErrorKo = 0;
	}
	void initMoKo(){//individual instantiation of the MP class
		prevInputs = new double*[numLayers];//for the final result
		deltas = new double*[numLayers];
		network = new double*[numLayers - 1];
		for(int q = 0; q < numLayers; q++){
			prevInputs[q] = new double[numNodesInLayer[q]];
			deltas[q] = new double[numNodesInLayer[q]];
		}
	}
	void backpropagate(double* finalDelta){
		deltas[numLayers-1] = finalDelta;
		//update the layer
		for(int q = numLayers - 1; q > 0; q--){
			//update the nodes in the layer
			for(int w = 0; w < numNodesInLayer[q]; w++){
				double eq1 = deltas[q][w];
				deltas[q-1][w] = 0;
				double eq2 = prevInputs[q][w];
				eq2 = eq2 * (1-eq2);
				//update the weights in each layer
				for(int e = 0; e < numNodesInLayer[q - 1]; e++){
					double eq3 = prevInputs[q - 1][e];
					deltas[q-1][w] += eq1 * eq2 * network[q - 1][(w * numNodesInLayer[q - 1]) + e];
					network[q - 1][(w * numNodesInLayer[q - 1]) + e] -= eq1 * eq2 * eq3 * learningRate;
				}
			}
		}
	}
	//numSavedInputs == numInputs
	double* feedForward(double* inputs){//returns array of activated values for each layer, 0 is for input->h1, 1 is for h1->h2, etc.
	//will save the prevInputs
		for(int w = 0; w < numNodesInLayer[0]; w++){
			prevInputs[0][w] = inputs[w];
		}
		//start processing the input->hidden layer
		for(int q = 0; q < numLayers - 1; q++){
			for(int w = 0; w < numNodesInLayer[q + 1]; w++){
				double activatedVal = 0;
				for(int e = 0; e < numNodesInLayer[q]; e++) activatedVal += network[q][e + (numNodesInLayer[q] * w)] * prevInputs[q][e];
				activatedVal += bias * biasWeight;
				//acitivatedVal is now the sum of the weighted inputs
				//run it through the activation function
				activatedVal = 1.0/(1.0 + exp(-1.0 * activatedVal));
				prevInputs[q + 1][w] = activatedVal;
			}
		}
		return prevInputs[numLayers - 1];
	}
};

MP** MPs;

void printWeights(ofstream* palabas){
	for(int q = 0; q < numLayers - 1; q++){
		(*palabas) << "In Layer " << q << endl;
		for(int w = 0; w < numNodesInLayer[q] * numNodesInLayer[q + 1]; w++){
			(*palabas) << globalNetwork[q][w] << endl;
		}
	}
}

void* processNode(void* arg){
	int index = (int) arg;
	//cout << "Got " << index << endl;
	MP *myMP = MPs[index];
	myMP->initMoKo();
	double* output, *delta, *error;
	int outputLength = numNodesInLayer[numLayers - 1];
	while(true){
		if(threadStates[index] == 0){
			//init the MPs first
			myMP->restart();//to get the global network
			int startingIndex = index * threadInterval;
			double iterationError = 0;
			for(int q = startingIndex; q < startingIndex + threadProcesses; q++){
				output = myMP->feedForward(globalInputs[q]);
				delta = new double[outputLength];
				error = new double[outputLength];
				//calculate the error rate, then run the backpropagation algorithm
				for(int w = 0; w < outputLength; w++){
					delta[w] = output[w] - globalOutputs[q][w];
					//cout << output[q] << " " << expectedOutput[q] << "\n";
				}
				//calculate for error
				
				for(int w = 0; w < outputLength; w++){
					error[w] = 0.5f * (globalOutputs[q][w] - output[w]) * (globalOutputs[q][w] - output[w]);
					iterationError += error[w];
				}
				myMP->backpropagate(delta);
			}
			iterationError = iterationError/threadProcesses;
			myMP->iterationErrorKo = iterationError;
			threadStates[index] = 1;
		}
		usleep(100);
	}
}

int main(){
	int iterationNumbers, inputLength, outputLength, numSets;
	double errorWindow;
	MP* angMPs;
	ifstream cinKo;
	ofstream palabas;
	palabas.open("Multi-Output.txt");
	cinKo.open("TrainingInputs.txt", ifstream::in);
	cinKo >> threadLength;
	cinKo >> inputLength;
	cinKo >> outputLength;
	cinKo >> errorWindow;
	cinKo >> iterationNumbers;
	cinKo >> numSets;

	threadInterval = (int)(numSets / threadLength);
	threadProcesses = (int)(numSets / threadLength) + (numSets % threadLength == 0 ? 0 : 1);

	MPs = new MP*[threadLength];
	angMPs = new MP[threadLength];
	angMPs[0].initOnce();//for instantiation of MP's 'static' variables
	globalInputs = new double*[numSets];
	globalOutputs = new double*[numSets];
	
	for(int w = 0; w < numSets; w++){
		globalInputs[w] = new double[inputLength];
		globalOutputs[w] = new double[outputLength];
		for(int q = 0; q < inputLength; q++){
			cinKo >> globalInputs[w][q];
		}
		
		for(int q = 0; q < outputLength; q++){
			cinKo >> globalOutputs[w][q];
		}
	}

	threadArray = new pthread_t[threadLength];
	threadStates = new int[threadLength];
	int returnVal = 0;
	for(int q = 0; q < threadLength; q++){
		threadStates[q] = 1;//1 para hindi sila agad tumakbo
		MPs[q] = &angMPs[q];
		returnVal = pthread_create(&threadArray[q], NULL, processNode, (void*)q);
	}
	
	//cout << "Running MP with inputs as " << inputs[0] << ", " << inputs[1] << "\n";
	//cout << "with ideal output as " << expectedOutput[0] << ", " << expectedOutput[1] << ", " << expectedOutput[2] << "\n";
	palabas << "Initial Weights: " << endl;
	printWeights(&palabas);
	auto start = chrono::steady_clock::now();
	for(int e = 0; e < iterationNumbers; e++){
		//start the threads
		for(int q = 0; q < threadLength; q++){
			threadStates[q] = 0;
		}

		waitForThreads();

		//consolidate the weights generated
		for(int q = 0; q < numLayers - 1; q++){
			int numNodesNow = numNodesInLayer[q] * numNodesInLayer[q + 1];
			for(int w = 0; w < numNodesNow; w++){
				double theNewWeight = 0;
				for(int r = 0; r < threadLength; r++){
					theNewWeight += angMPs[r].network[q][w];
				}
				theNewWeight = theNewWeight/threadLength;
				globalNetwork[q][w] = theNewWeight;
			}
		}
	}
	
	palabas << "Final Weights: " << endl;
	printWeights(&palabas);
	auto end = chrono::steady_clock::now();
	auto diff = end - start;
	palabas << "Elapsed time is " << chrono::duration_cast<chrono::nanoseconds>(diff).count() << " ns \n";
	//add the validation part of the set
	angMPs[0].restart();
	palabas << "Done Training\n";
	palabas << "Processing Validation Set\n";
	cinKo.close();
	cinKo.open("ValidationInputs.txt", ifstream::in);
	int numValRecur;
	double errorInRecur = 0;
	cinKo >> numValRecur;
	int numCorrect = 0;
	for(int q = 0; q < numValRecur; q++){
		double iterationError = 0;
		double* valInputs = new double[inputLength];
		double* valOutputs = new double[outputLength];
		for(int w = 0; w < inputLength; w++){
			cinKo >> valInputs[w];
		}
		for(int w = 0; w < outputLength; w++){
			cinKo >> valOutputs[w];
		}
		double* outputKo = angMPs[0].feedForward(valInputs);
		for(int w = 0; w < outputLength; w++){
			errorInRecur = sqrt((valOutputs[w] - outputKo[w]) * (valOutputs[w] - outputKo[w]));
			iterationError += errorInRecur;
		}
		string ilalabas;
		if(iterationError < errorWindow){
			ilalabas = "correct";
			numCorrect++;
		}
		else ilalabas = "wrong";
		palabas << "Validation Set# " << q + 1 << " - " << ilalabas << " with error value " << iterationError << " with activation value: " << outputKo[0] << endl;
	}	
	palabas << "Network Accuracy: " << (float)numCorrect/numValRecur << endl;
	palabas.close();
}