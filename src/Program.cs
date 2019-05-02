using System;
using System.Collections.Generic;
using System.Linq;

namespace src
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            var nn = new NeuralNetwork(21.9, new int [] { 2, 4, 2});


            var input = new List<double>() { 0.0 , 1.0 };
            var expectedOut = new List<double> { 1.0, 0.0 };
            for (int i = 0; i < 100; i++) {
                nn.Train(input, expectedOut);
            }
            
            var output = nn.Run(input);
            

            Console.WriteLine(output[0]);
            Console.WriteLine(output[1]);

            int layerid = 0;
            foreach ( var layer in nn.Layers) {
                Console.WriteLine($"layer {layerid++}");
                foreach (var neuron in layer.Neurons) {
                    Console.WriteLine(neuron.Value);
                }
            }
        }
    }
}
