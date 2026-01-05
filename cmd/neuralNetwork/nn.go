package main

import (
	"fmt"
	"math"
	"os"

	"github.com/arnicfil/go_neural_network/internal/matrixHelpers"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type NeuralNetwork struct {
	inputSize     int
	hiddenSize    int
	outputSize    int
	hiddenWeights *mat.Dense
	outputWeights *mat.Dense
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

func sigmoid(r, c int, x float64) float64 {
	return 1 / (1 + math.Exp(-1*x))
}

func sigmoidPrime(r, c int, x float64) float64 {
	val := sigmoid(r, c, x)
	return val * (1 - val)
}

func (n NeuralNetwork) Predict(inputData []float64) (mat.Matrix, mat.Matrix, error) {
	inputs := mat.NewDense(len(inputData), 1, inputData)

	hiddenInputs, err := matrixhelpers.Dot(n.hiddenWeights, inputs)
	if err != nil {
		return nil, nil, fmt.Errorf("Error calculating hiddeninputs: %v", err)
	}
	hiddenOutputs := matrixhelpers.ApplyFunction(sigmoid, hiddenInputs)

	finalOutputs, err := matrixhelpers.Dot(n.outputWeights, hiddenOutputs)
	if err != nil {
		return nil, nil, fmt.Errorf("Error calculating finalOutputs: %v", err)
	}
	finalOutputs = matrixhelpers.ApplyFunction(sigmoid, finalOutputs)

	return finalOutputs, hiddenOutputs, nil
}

func (n *NeuralNetwork) Train(inputData []float64, targetData []float64) error {
	inputs := mat.NewDense(len(inputData), 1, inputData)
	targets := mat.NewDense(len(targetData), 1, targetData)

	networkOutput, hiddenOuputs, err := n.Predict(inputData)
	if err != nil {
		return fmt.Errorf("Error in predicting: %v", err)
	}

	//Calculate the error of the final layer
	outputLayerError, err := matrixhelpers.Subtract(targets, networkOutput)
	if err != nil {
		return fmt.Errorf("Error calculating outputLayerError: %v", err)
	}
	//Take the derivative of the final layer
	finalLayerSlope := matrixhelpers.ApplyFunction(sigmoidPrime, networkOutput)
	//Adjust the error based on how bad it is
	adjustedFinalLayerError, err := matrixhelpers.Multiply(outputLayerError, finalLayerSlope)
	if err != nil {
		return fmt.Errorf("Error calculating adjustedFinalLayerError: %v", err)
	}
	finalLayerChange, err := matrixhelpers.Dot(adjustedFinalLayerError, hiddenOuputs.T())
	if err != nil {
		return fmt.Errorf("Error calculating finalLayerChange: %v", err)
	}

	hiddenLayerError, err := matrixhelpers.Dot(n.outputWeights.T(), outputLayerError)
	if err != nil {
		return fmt.Errorf("Error calculating hiddenLayerError: %v", err)
	}
	hiddenLayerSlope := matrixhelpers.ApplyFunction(sigmoidPrime, hiddenOuputs)
	adjustedHiddenLayerError, err := matrixhelpers.Multiply(hiddenLayerError, hiddenLayerSlope)
	if err != nil {
		return fmt.Errorf("Error adjustedHiddenLayerError: %v", err)
	}
	hiddenLayerChange, err := matrixhelpers.Dot(adjustedHiddenLayerError, inputs.T())
	if err != nil {
		return fmt.Errorf("Error calculating hiddenLayerChange: %v", err)
	}

	n.outputWeights, err = matrixhelpers.Add(n.outputWeights, matrixhelpers.Scale(n.learningRate, finalLayerChange))
	if err != nil {
		return fmt.Errorf("Error calculating new outputWeights: %v", err)
	}

	n.hiddenWeights, err = matrixhelpers.Add(n.hiddenWeights, matrixhelpers.Scale(n.learningRate, hiddenLayerChange))
	if err != nil {
		return fmt.Errorf("Error calculating new hiddenWeights: %v", err)
	}

	return nil
}

func (n NeuralNetwork) save(fileName string) error {
	h, err := os.Create("data/h" + fileName)
	if err != nil {
		return fmt.Errorf("Error creating hidden weights file: %v", err)
	}
	defer h.Close()

	_, err = n.hiddenWeights.MarshalBinaryTo(h)
	if err != nil {
		return fmt.Errorf("Error marshaling hidden weights:  %v", err)
	}

	o, err := os.Create("data/o" + fileName)
	if err != nil {
		return fmt.Errorf("Error creating output weights file: %v", err)
	}
	defer o.Close()

	_, err = n.outputWeights.MarshalBinaryTo(o)
	if err != nil {
		return fmt.Errorf("Error marshaling output weights: %v", err)
	}

	return nil
}

func (n *NeuralNetwork) load(fileName string) error {
	h, err := os.Open("data/h" + fileName)
	if err != nil {
		return fmt.Errorf("Error opening hidden weights file: %v", err)
	}
	defer h.Close()

	n.hiddenWeights.Reset()
	_, err = n.hiddenWeights.UnmarshalBinaryFrom(h)
	if err != nil {
		return fmt.Errorf("Error unmarshaling hidden weights:  %v", err)
	}

	o, err := os.Open("data/o" + fileName)
	if err != nil {
		return fmt.Errorf("Error opening output weights file: %v", err)
	}
	defer o.Close()

	n.outputWeights.Reset()
	_, err = n.outputWeights.UnmarshalBinaryFrom(o)
	if err != nil {
		return fmt.Errorf("Error unmarshaling output weights: %v", err)
	}

	return nil
}
