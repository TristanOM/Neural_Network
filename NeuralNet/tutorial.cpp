	#include <vector>
	#include <iostream>
	#include <cstdlib>
	#include <cassert>
	#include <cmath>

	using namespace std;

	struct Connection
	{
		double weight;
		double deltaWeight;
	};

	class Neuron;

	typedef vector<Neuron> Layer;


	//________________________________________________________________
	class Neuron
	{
	public:
		Neuron(unsigned numOutputs, unsigned myIndex);
		void setOutputVal(double val){m_outputVal = val;}
		double getOutputVal(void) const {return m_outputVal;}
		void feedForward(const Layer &prevLayer);
		void calcOutputGradients(double targetVal);
		void calcHiddenGradients(const Layer &nextLayer);
		void updateInputWeights(Layer &prevLayer);
	private:
		static double eta;//0-1
		static double alpha; //0-n
		static double transferFunction(double x);
		static double transferFunctionDerivative(double x);
		static double randomWeight(void){return rand() /double(RAND_MAX);};
		double sumDow(const Layer &nextLayer) const;
		double m_outputVal;
		vector <Connection> m_outputWeights;
		unsigned m_myIndex;
		double m_gradient;
	};

	double Neuron::eta = 0.15;
	double Neuron::alpha = 0.5;

	void Neuron::updateInputWeights(Layer &prevLayer)
	{
		//weights to be updates in connection container in the neurons in the preceding layer

		for(unsigned n =0; n <prevLayer.size(); ++n){
			Neuron &neuron = prevLayer[n];
			double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

			double newDeltaWeight =
				// Individual input, magnified by gradient and train rate
				eta
				*neuron.getOutputVal()
				*m_gradient
				//also momentum
				+alpha
				*oldDeltaWeight;

			neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
			neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
		}

	}

	double Neuron::sumDow(const Layer &nextLayer) const
	{
		double sum = 0.0;

		//Sum our contributions of errors at the nodes we feed

		for(unsigned n = 0; n <nextLayer.size() -1; ++n){
				sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
		}

		return sum;
	}

	void Neuron::calcHiddenGradients(const Layer &nextLayer)
	{
		double dow = sumDow(nextLayer);
		m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
	}

	void Neuron::calcOutputGradients(double targetVal)
	{
		double delta = targetVal - m_outputVal;
		m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
	}

	double Neuron::transferFunction(double x){
		return tanh(x);
	}

	double Neuron::transferFunctionDerivative(double x){
		return (1.0 - x*x);
	}

	void Neuron::feedForward(const Layer &prevLayer)
	{
		double sum = 0.0;

		for(unsigned n = 0; n <prevLayer.size(); ++n)
		{
			sum += prevLayer[n].getOutputVal()* prevLayer[n].m_outputWeights[m_myIndex].weight;
		}

		m_outputVal = transferFunction(sum);
	}

	Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
	{
		for(unsigned c = 0; c<numOutputs; ++c)
		{
			m_outputWeights.push_back(Connection());
			m_outputWeights.back().weight = randomWeight();
		}

		m_myIndex = myIndex;
	}
	//________________________________________________________________
	class Net
	{
	public:
		Net(const vector<unsigned> &topology);
		void feedForward(const vector<double> &inputVals);
		void backProp(const vector<double> &targetVals);
		void getResults(vector<double> &resultVals) const;
		double getError(const vector<double> &targetVals);

	private:
		vector<Layer> m_layers;
		double m_error;
		double m_recentAverageError;
		double m_recentAverageSmoothingFactor;

	};

	double Net::getError(const vector<double> &targetVals)
	{
		Layer &outputLayer = m_layers.back();
		double error = 0.0;

		for(unsigned n = 0; n<outputLayer.size() -1; ++n){
			double delta = targetVals[n] - outputLayer[n].getOutputVal();
			error +=delta *delta;
		}
		error /= outputLayer.size() -1;//avg erro sqaured
		error = sqrt(m_error);// RMS

		
		return error;
	}

	void Net::getResults(vector<double> &resultVals) const
	{
		resultVals.clear();

		for(unsigned n = 0; n < m_layers.back().size() -1; ++n){
			resultVals.push_back(m_layers.back()[n].getOutputVal());
			cout << "result " << m_layers.back()[n].getOutputVal() << endl;//print
		}
	}

	void Net::backProp(const vector<double> &targetVals)
	{
		// Caculate overalll net error (RMS of outputs neruon errors)

		Layer &outputLayer = m_layers.back();
		m_error = 0.0;

		for(unsigned n = 0; n<outputLayer.size() -1; ++n){
			double delta = targetVals[n] - outputLayer[n].getOutputVal();
			m_error +=delta *delta;
		}
		m_error /= outputLayer.size() -1;//avg erro sqaured
		m_error = sqrt(m_error);// RMS

		cout << "error " << m_error << endl;//print

		//implement a recent average measurement

		m_recentAverageError = 
			(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)/(m_recentAverageSmoothingFactor +1.0);
		
		//cout << "recent average error " << m_recentAverageError << endl;//print
		//Calc output layer gradients

		for(unsigned n = 0; n <outputLayer.size() -1; ++n){
			outputLayer[n].calcOutputGradients(targetVals[n]);
		}

		//Calc gradients on hidden layers

		for(unsigned layerNum = m_layers.size() -2; layerNum >0; --layerNum){
			Layer &hiddenLayer = m_layers[layerNum];
			Layer &nextLayer = m_layers[layerNum +1];

			for(unsigned n =0; n< hiddenLayer.size(); ++n){
				hiddenLayer[n].calcHiddenGradients(nextLayer);
			}
		}

		//For all layer form outputs to first hidden layer
		//update connection weights

		for(unsigned layerNum = m_layers.size() -1; layerNum > 0; --layerNum){
			Layer &layer = m_layers[layerNum];
			Layer &prevLayer = m_layers[layerNum - 1];

			for(unsigned n = 0; n <layer.size() -1; ++n){
				layer[n].updateInputWeights(prevLayer);
			}
		}
	}

	void Net::feedForward(const vector<double> &inputVals)
	{
		assert(inputVals.size() == m_layers[0].size() -1);

		for(unsigned i =0; i <inputVals.size(); i++)
		{
			m_layers[0][i].setOutputVal(inputVals[i]);

		}

		//Forward propagation

		for(unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
		{
			Layer &prevLayer = m_layers[layerNum -1];
			for(unsigned n = 0; n < m_layers[layerNum].size() -1; ++n)
			{
				m_layers[layerNum][n].feedForward(prevLayer);
			}
		}

	}

	Net::Net(const vector<unsigned> &topology)
	{
		unsigned numLayers = topology.size();

		for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
		{
			m_layers.push_back(Layer());
			unsigned numOutputs = layerNum == topology.size() -1 ? 0 : topology[layerNum + 1];
			for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
			{
				m_layers.back().push_back(Neuron(numOutputs, neuronNum));
				cout <<"New Neuron"<<endl;
			}
			m_layers.back().back().setOutputVal(1.0);
		}
	}

	//________________________________________________________________

	int testBasic(vector<double> &inputVals){
		double Random = rand() /double(RAND_MAX);
		double Random2 = rand() /double(RAND_MAX);
		inputVals.clear();
		if(Random < .5){
			inputVals.push_back(1);
		}
		else
			inputVals.push_back(0);
		if(Random2 < .5){
			inputVals.push_back(1);
		}
		else
			inputVals.push_back(0);

		if(inputVals[0] == inputVals[1])
			return 0;
		else
			return 1;
	}


	double test1d(vector<double> &inputVals){
		double Random = rand() /double(RAND_MAX);
		inputVals.clear();
		
		inputVals.push_back(Random);
			return (0.5 - Random);
	}

	double test3d(vector<double> &inputVals){
		double RandomUD = rand() /double(RAND_MAX);
		double RandomFB = rand() /double(RAND_MAX);
		double RandomLR = rand() /double(RAND_MAX);
		inputVals.clear();
		
		inputVals.push_back(RandomUD);
		inputVals.push_back(RandomFB);
		inputVals.push_back(RandomLR);

		double length = sqrt((RandomUD * RandomUD) + (RandomFB * RandomFB) + (RandomLR * RandomLR));
		if (length < .5 )
			return 0;
		else
			return 1;

			//return (0.5 - Random);
	}

	int main()
	{
		vector<unsigned> topology;
		vector<double> resultVals;
		vector<double> inputVals;
		vector<double> target;

		topology.push_back(3);
		topology.push_back(3);
		topology.push_back(3);
		topology.push_back(33);
		topology.push_back(1);

		Net myNet(topology);

		
		int a = 1;
		double avgError = 0.0;

		while(a!=0){
			for(int i =0; i<100;i++){
				target.clear();
				//target.push_back (test1d(inputVals));
				target.push_back (test3d(inputVals));
				cout<< "Input: "<< inputVals[0] <<endl;
				cout<< "Target: "<< target[0]<<endl;
				myNet.feedForward(inputVals);
				myNet.getResults(resultVals);
				myNet.backProp(target);

				avgError = avgError + myNet.getError(target);
			}
			cout <<"Average Error for 100 " << avgError/100 <<endl;
			avgError = 0.0;
		
			cout << "100 more? 0 to stop"<<endl;
			cin >> a;
		}


		/*while(a!=0){
			for(int i =0; i<100;i++){
				target.clear();
				target.push_back (testBasic(inputVals));
				cout<< "Input: "<< inputVals[0] <<" "<< inputVals[1]<<endl;
				cout<< "Target: "<< target[0]<<endl;
				myNet.feedForward(inputVals);
				myNet.getResults(resultVals);
				myNet.backProp(target);
			}
		
			cout << "100 more? 0 to stop"<<endl;
			cin >> a;
		}
		*/
	
		cin >> a ;
	}