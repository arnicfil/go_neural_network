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
	inputs := mat.NewDense(len(targetData), 1, inputData)
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
	data, err := n.hiddenWeights.MarshalBinary()
	if err != nil {
		return fmt.Errorf("Error marshaling hiddenWeigts: %v", err)
	}
	err = os.WriteFile("data/h"+fileName, data, 0644)
	if err != nil {
		return fmt.Errorf("Error writing file: %v", err)
	}

	data, err = n.outputWeights.MarshalBinary()
	if err != nil {
		return fmt.Errorf("Error marshaling outputWeights: %v", err)
	}
	err = os.WriteFile("data/o"+fileName, data, 0644)
	if err != nil {
		return fmt.Errorf("Error writing file: %v", err)
	}

	return nil
}

func (n *NeuralNetwork) load(fileName string) error {
	data, err := os.ReadFile("data/h" + fileName)
	if err != nil {
		return fmt.Errorf("Error reading file: %v", err)
	}

	n.hiddenWeights = &mat.Dense{}
	err = n.hiddenWeights.UnmarshalBinary(data)
	if err != nil {
		return fmt.Errorf("Error unmarshaling data: %v", err)
	}

	data, err = os.ReadFile("data/o" + fileName)
	if err != nil {
		return fmt.Errorf("Error reading file: %v", err)
	}

	n.hiddenWeights = &mat.Dense{}
	err = n.outputWeights.UnmarshalBinary(data)
	if err != nil {
		return fmt.Errorf("Error unmarshaling data: %v", err)
	}

	return nil
}
