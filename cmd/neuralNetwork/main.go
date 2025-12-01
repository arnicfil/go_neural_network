package main

import (
	"math"
	"github.com/arnicfil/go_neural_network"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type NeuralNetwork struct {
	inputSize     int
	hiddenSize    int
	outputSize    int
	hiddenWeights mat.Matrix
	outputWeights mat.Matrix
	learningRate  float64
}

func randomArray(size int, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}

	data = make([]float64, size)
	for i := range data {
		data[i] = dist.Rand()
	}
	return data
}

func newNetwork(inputSize, hiddenSize, outputSize int, learningRate float64) NeuralNetwork {
	n := NeuralNetwork{
		inputSize:    inputSize,
		hiddenSize:   hiddenSize,
		outputSize:   outputSize,
		learningRate: learningRate,
	}

	n.hiddenWeights = mat.NewDense(n.hiddenSize, n.inputSize, randomArray(n.hiddenSize*n.inputSize, float64(n.inputSize)))
	n.outputWeights = mat.NewDense(n.outputSize, n.hiddenSize, randomArray(n.outputSize*n.hiddenSize, float64(n.hiddenSize)))
	return n
}

func (n NeuralNetwork) Predict(inputData []float64) mat.Matrix {
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := 
}
