
using System; 
namespace FeedForward {
    class FeedForwardProgram {
        static void Main(string[] args) {
            Console.WriteLine("\nBegin feed-forward demo\n");
            int numInput = 3; int numHidden = 4; int numOutput = 2;
            Console.WriteLine("Creating a 3-4-2 tanh-softmax neural network"); NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);
            double[] weights = new double[] { 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
                0.09, 0.10, 0.11, 0.12,
                0.13, 0.14, 0.15, 0.16,
                0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26 };
            Console.WriteLine("\nSetting dummy weights and biases:"); ShowVector(weights, 8, 2, true);
            nn.SetWeights(weights);
            double[] xValues = new double[] { 1.0, 2.0, 3.0 }; Console.WriteLine("\nInputs are:"); ShowVector(xValues, 3, 1, true);
            Console.WriteLine("\nComputing outputs");

            double[] yValues = nn.ComputeOutputs(xValues); Console.WriteLine("\nOutputs computed");
            Console.WriteLine("\nOutputs are:"); ShowVector(yValues, 2, 4, true);
            Console.WriteLine("\nEnd feed-forward demo\n");
            Console.ReadLine(); } // Main
        public static void ShowVector(double[] vector, int valsPerRow, int decimals, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i) {
                if (i % valsPerRow == 0) Console.WriteLine("");
                Console.Write(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " "); }
            if (newLine == true) Console.WriteLine("");
        }
    } // Program
    public class NeuralNetwork {
        private int numInput; private int numHidden; private int numOutput;
        private double[] inputs;
        private double[][] ihWeights; private double[] hBiases; private double[] hOutputs;
        private double[][] hoWeights; private double[] oBiases; private double[] outputs;
        public NeuralNetwork(int numInput, int numHidden, int numOutput) {
            this.numInput = numInput; this.numHidden = numHidden; this.numOutput = numOutput;
            this.inputs = new double[numInput]; this.ihWeights = MakeMatrix(numInput, numHidden); this.hBiases = new double[numHidden]; this.hOutputs = new double[numHidden];
            this.hoWeights = MakeMatrix(numHidden, numOutput); this.oBiases = new double[numOutput];
            this.outputs = new double[numOutput];
        }
        private static double[][] MakeMatrix(int rows, int cols) {
            double[][] result = new double[rows][]; for (int i = 0; i < rows; ++i)
                result[i] = new double[cols];



            return result; 
        }

        public void SetWeights(double[] weights) {
            int numWeights = (numInput * numHidden) + numHidden + (numHidden * numOutput) + numOutput;
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array");
            int k = 0; // Pointer into weights.
            for (int i = 0; i < numInput; ++i) for (int j = 0; j < numHidden; ++j)
                ihWeights[i][j] = weights[k++];
            for (int i = 0; i < numHidden; ++i) hBiases[i] = weights[k++];
            for (int i = 0; i < numHidden; ++i) for (int j = 0; j < numOutput; ++j)
                hoWeights[i][j] = weights[k++];
            for (int i = 0; i < numOutput; ++i) oBiases[i] = weights[k++];
        }

        public double[] ComputeOutputs(double[] xValues) {
            if (xValues.Length != numInput)
                throw new Exception("Bad xValues array");
            double[] hSums = new double[numHidden]; double[] oSums = new double[numOutput];
            for (int i = 0; i < xValues.Length; ++i) inputs[i] = xValues[i];
            // ex: hSum[0] = (in[0] * ihW[[0][0]) + (in[1] * ihW[1][0]) + (in[2] * ihW[2][0]) + . . // hSum[1] = (in[0] * ihW[[0][1]) + (in[1] * ihW[1][1]) + (in[2] * ihW[2][1]) + . . // ...
            for (int j = 0; j < numHidden; ++j) for (int i = 0; i < numInput; ++i)
                hSums[j] += inputs[i] * ihWeights[i][j];
            for (int i = 0; i < numHidden; ++i) hSums[i] += hBiases[i];
            Console.WriteLine("\nPre-activation hidden sums:"); FeedForwardProgram.ShowVector(hSums, 4, 4, true);
            for (int i = 0; i < numHidden; ++i) hOutputs[i] = HyperTan(hSums[i]);
            Console.WriteLine("\nHidden outputs:"); FeedForwardProgram.ShowVector(hOutputs, 4, 4, true);
            for (int j = 0; j < numOutput; ++j) for (int i = 0; i < numHidden; ++i)
                oSums[j] += hOutputs[i] * hoWeights[i][j];


            for (int i = 0; i < numOutput; ++i) oSums[i] += oBiases[i];
            Console.WriteLine("\nPre-activation output sums:"); FeedForwardProgram.ShowVector(oSums, 2, 4, true);
            double[] softOut = Softmax(oSums); // Softmax does all outputs at once. 
            for (int i = 0; i < outputs.Length; ++i)
                outputs[i] = softOut[i];
            
            double[] result = new double[numOutput]; 

            for (int i = 0; i < outputs.Length; ++i)
                result[i] = outputs[i];
            return result; 
        }

        private static double HyperTan(double v) {
            if (v < -20.0) return -1.0;
            else if (v > 20.0) return 1.0;
            else
                return Math.Tanh(v);
        }
        public static double[] Softmax(double[] oSums) {
            // Does all output nodes at once.
            // Determine max oSum.
            double max = oSums[0];
            for (int i = 0; i < oSums.Length; ++i)
                if (oSums[i] > max) max = oSums[i];
            // Determine scaling factor -- sum of exp(each val - max).
            double scale = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
                scale += Math.Exp(oSums[i] - max);
            double[] result = new double[oSums.Length]; for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i] - max) / scale; return result; // Now scaled so that xi sums to 1.0.
        }
        public static double[] SoftmaxNaive(double[] oSums) {
            double denom = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
                denom += Math.Exp(oSums[i]);
            double[] result = new double[oSums.Length]; for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i]) / denom; return result;
        }
    } // NeuralNetwork
} // ns

