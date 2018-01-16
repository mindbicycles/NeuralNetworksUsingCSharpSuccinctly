using System; namespace BackProp {
    class BackPropProgram {
        static void Main(string[] args) {
            Console.WriteLine("\nBegin back-propagation demo\n");
            Console.WriteLine("Creating a 3-4-2 neural network\n"); int numInput = 3;
            int numHidden = 4;
            int numOutput = 2;

            NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);
            double[] weights = new double[26] {
                0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26 };
            Console.WriteLine("Setting dummy initial weights to:"); 
            ShowVector(weights, 8, 2, true); 
            nn.SetWeights(weights);
            double[] xValues = new double[3] { 1.0, 2.0, 3.0 }; 
            double[] tValues = new double[2] { 0.2500, 0.7500 }; // Target outputs.

            Console.WriteLine("\nSetting fixed inputs = "); 
            ShowVector(xValues, 3, 1, true); 
            Console.WriteLine("Setting fixed target outputs = "); 
            ShowVector(tValues, 2, 4, true);

            double learnRate = 0.05;
            double momentum = 0.01;
            int maxEpochs = 1000;
            Console.WriteLine("\nSetting learning rate = " + learnRate.ToString("F2")); 
            Console.WriteLine("Setting momentum = " + momentum.ToString("F2")); 
            Console.WriteLine("Setting max epochs = " + maxEpochs + "\n");
            nn.FindWeights(tValues, xValues, learnRate, momentum, maxEpochs);
            double[] bestWeights = nn.GetWeights(); Console.WriteLine("\nBest weights found:"); 
            ShowVector(bestWeights, 8, 4, true);
            Console.WriteLine("\nEnd back-propagation demo\n");
            Console.ReadLine(); 
        } // Main

        public static void ShowVector(double[] vector, int valsPerRow, int decimals, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i) {
                if (i > 0 && i % valsPerRow == 0) Console.WriteLine("");
                Console.Write(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " "); 
            }
            if (newLine == true) Console.WriteLine(""); 
        }

        public static void ShowMatrix(double[][] matrix, int decimals) 
        {
            int cols = matrix[0].Length;
            for (int i = 0; i < matrix.Length; ++i) // Each row.
                ShowVector(matrix[i], cols, decimals, true);
        }

    } 

    // Program class
    public class NeuralNetwork
    {
        private int numInput; 
        private int numHidden; 
        private int numOutput;
        private double[] inputs;
        private double[][] ihWeights; 
        private double[] hBiases; 
        private double[] hOutputs;
        private double[][] hoWeights; 
        private double[] oBiases; 
        private double[] outputs;
        private double[] oGrads; // Output gradients for back-propagation.
        private double[] hGrads; // Hidden gradients for back-propagation.
        private double[][] ihPrevWeightsDelta; // For momentum. 
        private double[] hPrevBiasesDelta;
        private double[][] hoPrevWeightsDelta;
        private double[] oPrevBiasesDelta;

        public NeuralNetwork(int numInput, int numHidden, int numOutput) {
            this.numInput = numInput; 
            this.numHidden = numHidden; 
            this.numOutput = numOutput;
            this.inputs = new double[numInput]; 
            this.ihWeights = MakeMatrix(numInput, numHidden); 
            this.hBiases = new double[numHidden]; 
            this.hOutputs = new double[numHidden];
            this.hoWeights = MakeMatrix(numHidden, numOutput); 
            this.oBiases = new double[numOutput]; 
            this.outputs = new double[numOutput];
            oGrads = new double[numOutput]; 
            hGrads = new double[numHidden];
            ihPrevWeightsDelta = MakeMatrix(numInput, numHidden); 
            hPrevBiasesDelta = new double[numHidden]; 
            hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput); 
            oPrevBiasesDelta = new double[numOutput];
            InitMatrix(ihPrevWeightsDelta, 0.011); 
            InitVector(hPrevBiasesDelta, 0.011); 
            InitMatrix(hoPrevWeightsDelta, 0.011); 
            InitVector(oPrevBiasesDelta, 0.011);
        }

        private static double[][] MakeMatrix(int rows, int cols) 
        {
            double[][] result = new double[rows][]; 
            for (int i = 0; i < rows; ++i)
                result[i] = new double[cols]; 

            return result;
        }

        private static void InitMatrix(double[][] matrix, double value) {
            int rows = matrix.Length;
            int cols = matrix[0].Length; 
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j) 
                    matrix[i][j] = value;
        }

        private static void InitVector(double[] vector, double value) 
        {
            for (int i = 0; i < vector.Length; ++i) 
                vector[i] = value;
        }

        public void SetWeights(double[] weights) 
        {
            //int numWeights = (numInput * numHidden) + numHidden + // (numHidden * numOutput) + numOutput;
            //if (weights.Length != numWeights)
            // throw new Exception("Bad weights array");
            int k = 0; // Pointer into weights.
            for (int i = 0; i < numInput; ++i) 
                for (int j = 0; j < numHidden; ++j)
                ihWeights[i][j] = weights[k++];
            
            for (int i = 0; i < numHidden; ++i) 
                hBiases[i] = weights[k++];
            
            for (int i = 0; i < numHidden; ++i) 
                for (int j = 0; j < numOutput; ++j)
                hoWeights[i][j] = weights[k++];
            
            for (int i = 0; i < numOutput; ++i) 
                oBiases[i] = weights[k++];
        }

        public double[] GetWeights() 
        {
            int numWeights = (numInput * numHidden) + numHidden + (numHidden * numOutput) + numOutput;
            double[] result = new double[numWeights];
            int k = 0;
            for (int i = 0; i < numInput; ++i)
                for (int j = 0; j < numHidden; ++j) 
                    result[k++] = ihWeights[i][j];
            
            for (int i = 0; i < numHidden; ++i) 
                result[k++] = hBiases[i];
            
            for (int i = 0; i < numHidden; ++i) 
                for (int j = 0; j < numOutput; ++j)
                result[k++] = hoWeights[i][j];
            
            for (int i = 0; i < numOutput; ++i) 
                result[k++] = oBiases[i];
            return result; 
        }

        private double[] ComputeOutputs(double[] xValues)
        {
            if (xValues.Length != numInput)
                throw new Exception("Bad xValues array");
            
            double[] hSums = new double[numHidden]; 
            double[] oSums = new double[numOutput];
            for (int i = 0; i < xValues.Length; ++i) 
                inputs[i] = xValues[i];
            
            for (int j = 0; j < numHidden; ++j) 
                for (int i = 0; i < numInput; ++i)
                hSums[j] += inputs[i] * ihWeights[i][j];
            
            for (int i = 0; i < numHidden; ++i) 
                hSums[i] += hBiases[i];
            
            for (int i = 0; i < numHidden; ++i) 
                hOutputs[i] = HyperTan(hSums[i]);
            
            for (int j = 0; j < numOutput; ++j) 
                for (int i = 0; i < numHidden; ++i)
                oSums[j] += hOutputs[i] * hoWeights[i][j];
            
            for (int i = 0; i < numOutput; ++i) 
                oSums[i] += oBiases[i];

            double[] softOut = Softmax(oSums); // All outputs at once. 

            for (int i = 0; i < outputs.Length; ++i)
                outputs[i] = softOut[i];

            double[] result = new double[numOutput]; 

            for (int i = 0; i < outputs.Length; ++i)
                result[i] = outputs[i];

            return result; 
        }

        private static double HyperTan(double v) {
            if (v < -20.0) 
                return -1.0;
            else if (v > 20.0) 
                return 1.0;
            else 
                return Math.Tanh(v); 
        }

        private static double[] Softmax(double[] oSums) {
            double max = oSums[0];
            for (int i = 0; i < oSums.Length; ++i)
                if (oSums[i] > max) max = oSums[i];
            double scale = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
                scale += Math.Exp(oSums[i] - max);
            double[] result = new double[oSums.Length]; 
            for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i] - max) / scale; 
            return result; // xi sum to 1.0.
        }

        public void FindWeights(double[] tValues, double[] xValues, double learnRate, double momentum, int maxEpochs)
        {
            // Call UpdateWeights maxEpoch times.
            int epoch = 0;
            while (epoch <= maxEpochs)
            {
                double[] yValues = ComputeOutputs(xValues); 
                UpdateWeights(tValues, learnRate, momentum);

                //if (epoch % 100 == 0) 
                {
                    Console.Write("epoch = " + epoch.ToString().PadLeft(5) + " current outputs = ");
                    BackPropProgram.ShowVector(yValues, 2, 4, true); 
                }
                ++epoch;
                        
            } // Find loop.
                        
        }

                        
        private void UpdateWeights(double[] tValues, double learnRate, double momentum) 
        {
            // Assumes that SetWeights and ComputeOutputs have been called.
            if (tValues.Length != numOutput)
                throw new Exception("target values not same Length as output in UpdateWeights");

            // 1. Compute output gradients. Assumes softmax.
            for (int i = 0; i < oGrads.Length; ++i) {
                double derivative = (1 - outputs[i]) * outputs[i]; // Derivative of softmax is y(1- y).
                oGrads[i] = derivative * (tValues[i] - outputs[i]); // oGrad = (1 - O)(O) * (T-O)
            }

            // 2. Compute hidden gradients. Assumes tanh!
            for (int i = 0; i < hGrads.Length; ++i)
            {
                double derivative = (1 - hOutputs[i]) * (1 + hOutputs[i]); // f' of tanh is (1-y)(1+y).
                double sum = 0.0;
                for (int j = 0; j < numOutput; ++j) // Each hidden delta is the sum of numOutputterms.
                    sum += oGrads[j] * hoWeights[i][j]; // Each downstream gradient * outgoing weight.
                hGrads[i] = derivative * sum; // hGrad = (1-O)(1+O) * Sum(oGrads*oWts)
            }

            // 3. Update input to hidden weights.
            for (int i = 0; i < ihWeights.Length; ++i)
            {
                for (int j = 0; j < ihWeights[i].Length; ++j)
                {
                    double delta = learnRate * hGrads[j] * inputs[i]; 
                    ihWeights[i][j] += delta; // Update.
                    ihWeights[i][j] += momentum * ihPrevWeightsDelta[i][j]; // Add momentum factor.
                    ihPrevWeightsDelta[i][j] = delta; // Save the delta for next time.
                }
            }
            // 4. Update hidden biases.
            for (int i = 0; i < hBiases.Length; ++i)
            {
                double delta = learnRate * hGrads[i] * 1.0; // The 1.0 is a dummy value; it could be left out.
                hBiases[i] += delta;
                hBiases[i] += momentum * hPrevBiasesDelta[i];
                hPrevBiasesDelta[i] = delta; // Save delta.
            }

            // 5. Update hidden to output weights.
            for (int i = 0; i < hoWeights.Length; ++i)
            { for (int j = 0; j < hoWeights[i].Length; ++j) 
                {
                    double delta = learnRate * oGrads[j] * hOutputs[i]; 
                    hoWeights[i][j] += delta;
                    hoWeights[i][j] += momentum * hoPrevWeightsDelta[i][j];
                    hoPrevWeightsDelta[i][j] = delta; // Save delta.
                }
            }
            // 6. Update output biases.
            for (int i = 0; i < oBiases.Length; ++i)
            {
                double delta = learnRate * oGrads[i] * 1.0;
                oBiases[i] += delta;
                oBiases[i] += momentum * oPrevBiasesDelta[i];
                oPrevBiasesDelta[i] = delta; // Save delta.
            }
        } // UpdateWeights
    } // NN class
} // ns
