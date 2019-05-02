
using System.Collections.Generic;
using System;
using System.Linq;

namespace src
{

    public class NeuralNetwork
    {
        public List<Layer> Layers { get; set; }
        public double LearningRate { get; set; }

        public int LayerCount
        {
            get
            {
                return Layers.Count;
            }
        }

        public NeuralNetwork(double learningRate, int[] layers)
        {
            if (layers.Length < 2) return;

            this.LearningRate = learningRate;
            this.Layers = new List<Layer>();

            for (int i = 0; i < layers.Length; i++)
            {
                var layer = new Layer(layers[i]);

                this.Layers.Add(layer);

                for (int neuron = 0; neuron < layers[i]; neuron++)
                {
                    layer.Neurons.Add(new Neuron());
                }

                foreach (var neuron in layer.Neurons)
                {
                    if (i == 0)
                    {
                        neuron.Bias = 0;
                    }
                    else
                    {
                        for (int dendrite = 0; dendrite < layers[i - 1]; dendrite++)
                        {
                            neuron.Dendrites.Add(new Dendrite());
                        }
                    }
                }
            }
        }

        public double[] Run(List<double> input)
        {
            if (input.Count != this.Layers[0].NeuronCount)
            {
                return null;
            }

            for (int i = 0; i < this.Layers.Count; i++)
            {
                Layer layer = this.Layers[i];

                for (int n = 0; n < layer.Neurons.Count; n++)
                {
                    var neuron = layer.Neurons[n];

                    if (i == 0)
                    {
                        neuron.Value = input[n];
                    }
                    else
                    {
                        neuron.Value = 0;
                        for (int np = 0; np < this.Layers[i - 1].Neurons.Count; np++)
                        {
                            neuron.Value = neuron.Value + this.Layers[i - 1].Neurons[np].Value * neuron.Dendrites[np].Weight;
                        }

                        neuron.Value = Sigmoid(neuron.Value + neuron.Bias);
                    }
                }
            }

            Layer last = this.Layers[this.Layers.Count - 1];
            int numOutput = last.Neurons.Count;
            double[] output = new double[numOutput];
            for (int i = 0; i < last.Neurons.Count; i++)
            {
                output[i] = last.Neurons[i].Value;
            }

            return output;
        }

        public bool Train(List<double> input, List<double> output)
        {
            if ((input.Count != this.Layers[0].Neurons.Count) || (output.Count != this.Layers[this.Layers.Count - 1].Neurons.Count))
            {
                return false;
            }

            Run(input);

            for (int i = 0; i < this.Layers.Last().Neurons.Count; i++)
            {
                var neuron = this.Layers.Last().Neurons[i];

                neuron.Delta = neuron.Value * (1 - neuron.Value) * (output[i] - neuron.Value);

                for (int j = this.Layers.Count - 2; j >= 1; j--)
                {
                    for (int k = 0; k < this.Layers[j].Neurons.Count; k++)
                    {
                        var n = this.Layers[j].Neurons[k];

                        n.Delta = n.Value *
                                    (1 - n.Value) *
                                    this.Layers[j + 1].Neurons[i].Dendrites[k].Weight *
                                    this.Layers[j + 1].Neurons[i].Delta;
                    }
                }
            }

            for (int i = this.Layers.Count - 1; i >= 1; i--)
            {
                for (int j = 0; j < this.Layers[i].Neurons.Count; j++)
                {
                    var n = this.Layers[i].Neurons[j];
                    n.Bias = n.Bias + (this.LearningRate * n.Delta);

                    for (int k = 0; k < n.Dendrites.Count; k++)
                    {
                        n.Dendrites[k].Weight = n.Dendrites[k].Weight + (this.LearningRate * this.Layers[i - 1].Neurons[k].Value * n.Delta);
                    }
                }
            }

            return true;
        }

        private double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
    }
}