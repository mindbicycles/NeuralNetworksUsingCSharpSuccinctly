using System;

namespace Perceptrons {
    class PerceptronProgram {
        
        static void Main(string[] args) 
        {
            Console.WriteLine("\nBegin perceptron demo\n");
            Console.WriteLine("Predict liberal (-1)  or conservative (+1) from age, income");
            double[][] trainData = new double[8][]; 
            trainData[0] = new double[] { 1.5, 2.0, -1 };
            trainData[1] = new double[] { 2.0, 3.5, -1 };
            trainData[2] = new double[] { 3.0, 5.0, -1 };
            trainData[3] = new double[] { 3.5, 2.5, -1 };
            trainData[4] = new double[] { 4.5, 5.0, 1 };
            trainData[5] = new double[] { 5.0, 7.0, 1 };
            trainData[6] = new double[] { 5.5, 8.0, 1 };
            trainData[7] = new double[] { 6.0, 6.0, 1 };

            Console.WriteLine("\nThe training data is:\n"); 
            ShowData(trainData);

            Console.WriteLine("\nCreating perceptron"); 
            int numInput = 2;
            Perceptron p = new Perceptron(numInput);
            double alpha = 0.001; int maxEpochs = 100;
            Console.Write("\nSetting learning rate to " + alpha.ToString("F3")); 
            Console.WriteLine(" and maxEpochs to " + maxEpochs);
            Console.WriteLine("\nBegin training");
            double[] weights = p.Train(trainData, alpha, maxEpochs);
            Console.WriteLine("Training complete"); 

            Console.WriteLine("\nBest weights and bias found:"); 
            ShowVector(weights, 4, true);
            double[][] newData = new double[6][];
            newData[0] = new double[] { 3.0, 4.0 }; // Should be -1. 
            newData[1] = new double[] { 0.0, 1.0 }; // Should be -1. 
            newData[2] = new double[] { 2.0, 5.0 }; // Should be -1. 
            newData[3] = new double[] { 5.0, 6.0 }; // Should be 1. 
            newData[4] = new double[] { 9.0, 9.0 }; // Should be 1. 
            newData[5] = new double[] { 4.0, 6.0 }; // Should be 1.

            Console.WriteLine("\nPredictions for new people:\n"); 
            for (int i = 0; i < newData.Length; ++i)
            {
                Console.Write("Age, Income = "); 
                ShowVector(newData[i], 1, false); 
                int c = p.ComputeOutput(newData[i]); 
                Console.Write(" Prediction is "); 
                if (c == -1)
                    Console.WriteLine("(-1) liberal"); 
                else if (c == 1)
                    Console.WriteLine("(+1) conservative"); 
            }
                Console.WriteLine("\nEnd perceptron demo\n");
                Console.ReadLine(); 
        } // Main
                
        static void ShowData(double[][] trainData) 
        {
            int numRows = trainData.Length;
            int numCols = trainData[0].Length; 
            for (int i = 0; i < numRows; ++i) 
            {
                Console.Write("[" + i.ToString().PadLeft(2, ' ') + "] "); 
                for (int j = 0; j < numCols - 1; ++j)
                        Console.Write(trainData[i][j].ToString("F1").PadLeft(6)); 

                Console.WriteLine(" -> " + trainData[i][numCols - 1].ToString("+0;-0"));
            } 
        }

        static void ShowVector(double[] vector, int decimals, bool newLine) 
        {
            for (int i = 0; i < vector.Length; ++i) 
            {
                if (vector[i] >= 0.0) 
                    Console.Write(" "); // For sign.

                Console.Write(vector[i].ToString("F" + decimals) + " "); 
            }

            if (newLine == true) 
                Console.WriteLine("");
        }
    } // Program



    public class Perceptron {
        private int numInput; private double[] inputs; private double[] weights; private double bias; private int output; private Random rnd;
        public Perceptron(int numInput) {
            this.numInput = numInput;
            this.inputs = new double[numInput]; 
            this.weights = new double[numInput]; 
            this.rnd = new Random(0); 
            InitializeWeights();
        }

        private void InitializeWeights() {
            double lo = -0.01;
            double hi = 0.01;
            for (int i = 0; i < weights.Length; ++i)
                weights[i] = (hi - lo) * rnd.NextDouble() + lo;

            bias = (hi - lo) * rnd.NextDouble() + lo;
        }

        public int ComputeOutput(double[] xValues) {
            if (xValues.Length != numInput)
                throw new Exception("Bad xValues in ComputeOutput");
            
            for (int i = 0; i < xValues.Length; ++i) 
                this.inputs[i] = xValues[i];
            
            double sum = 0.0;
            for (int i = 0; i < numInput; ++i)
                sum += this.inputs[i] * this.weights[i]; 
            
            sum += this.bias;
            int result = Activation(sum);
            this.output = result;
            return result; 
        }

        private static int Activation(double v) {
            if (v >= 0.0) return +1;
            else
                return -1;
        }

        public double[] Train(double[][] trainData, double alpha, int maxEpochs) 
        {
            int epoch = 0;
            double[] xValues = new double[numInput]; //allocate empty array to be reused in each training loop
            int desired = 0;

            //generate ordered sequence, to be shuffled randomly later, on each loop
            int[] sequence = new int[trainData.Length]; 
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            //train many times (maxEpochs times) with the 8 training entries
            while (epoch < maxEpochs) 
            {
                Shuffle(sequence);

                //loop through the 8 training entries
                for (int i = 0; i < trainData.Length; ++i) 
                {
                    int idx = sequence[i];
                    Array.Copy(trainData[idx], xValues, numInput); //copy the 2 input values of this training entry into xValues
                    desired = (int)trainData[idx][numInput]; // copy the last 'column' of the training entry that is the desired result (-1 or +1)
                    int computed = ComputeOutput(xValues); //compute perceptron output for these 2 inputs
                    Update(computed, desired, alpha); // Modify weights and bias values
                } // for each data.
                ++epoch; 
            }
            double[] result = new double[numInput + 1]; 
            Array.Copy(this.weights, result, numInput); 
            result[result.Length - 1] = bias; // Last cell. 
            return result;
        } 

        // Train
        private void Shuffle(int[] sequence) 
        {
            for (int i = 0; i < sequence.Length; ++i) 
            {
                int r = rnd.Next(i, sequence.Length); int tmp = sequence[r];
                sequence[r] = sequence[i]; sequence[i] = tmp;
            } 
        }

        //Modify weights and bias values
        private void Update(int computed, int desired, double alpha) //alpha is how much to increase or decrease the weight (the weight 'step')
        {

            if (computed == desired) return; // We're good.
            int delta = computed - desired; // If computed > desired, delta is +. 
            for (int i = 0; i < this.weights.Length; ++i) // Each input-weight pair. 
            {
                if (computed > desired && inputs[i] >= 0.0) // Need to reduce weights. If computed is bigger than expected and input is a positive number, then the weight needs to be smaller, to produce a smaller output when multiplied by input
                            weights[i] = weights[i] - (alpha * delta * inputs[i]); // delta +, alpha +,input
                else if (computed > desired && inputs[i] < 0.0) // Need to reduce weights. i.e: if(10 > 6 && -2 < 0) the weight to produce this result is -5 because -2 * -5 = 10, but I wanted it to produce 6, so the weight I need is -3, so multiplied by an input of -2 will output 6. Therefore, I need to reduce the weight (from -5 to -3) 
                            weights[i] = weights[i] + (alpha * delta * inputs[i]); // delta +, alpha +,input
                else if (computed < desired && inputs[i] >= 0.0) // Need to increase weights. 
                            weights[i] = weights[i] - (alpha * delta * inputs[i]); // delta -, aplha +,input
                else if (computed < desired && inputs[i] < 0.0) // Need to increase weights. 
                            weights[i] = weights[i] + (alpha * delta * inputs[i]); // delta -, alpha +,input
                
                // Logically equivalent:
                //If (inputs[i] >= 0.0) // Either reduce or increase weights (depending on delta). 
                // weights[i] = weights[i] - (alpha * delta * inputs[i]);
                //else
                // weights[i] = weights[i] + (alpha * delta * inputs[i]);
                // Also equivalent if all input > 0, but not obvious. //weights[i] = weights[i] - (alpha * delta * inputs[i]);
            } // Each weight.
            bias = bias - (alpha * delta);
        }
    } // Perceptron
} // ns