package matrixhelpers

import (
	"errors"

	"gonum.org/v1/gonum/mat"
)

var ERROR_DIMS = errors.New("Error dimensions aren't right")

func Dot(m, n mat.Matrix) (mat.Matrix, error) {
	w, _ := m.Dims()
	_, h := n.Dims()

	if w != h {
		return nil, ERROR_DIMS
	}

	res := mat.NewDense(w, h, nil)
	res.Product(m, n)
	return res, nil
}

func Scale(scale float64, matrix mat.Matrix) mat.Matrix {
	w, h := matrix.Dims()
	res := mat.NewDense(w, h, nil)
	res.Scale(scale, matrix)
	return res
}

func Multiply(a, b mat.Matrix) (mat.Matrix, error) {
	w1, h1 := a.Dims()
	w2, h2 := b.Dims()

	if w1 != w2 || h1 != h2 {
		return nil, ERROR_DIMS
	}

	res := mat.NewDense(w1, h1, nil)
	res.MulElem(a, b)
	return res, nil
}

func Add(a, b mat.Matrix) (mat.Matrix, error) {
	w1, h1 := a.Dims()
	w2, h2 := b.Dims()

	if w1 != w2 || h1 != h2 {
		return nil, ERROR_DIMS
	}

	res := mat.NewDense(w1, h1, nil)
	res.Add(a, b)
	return res, nil
}

func Subtract(a, b mat.Matrix) (mat.Matrix, error) {
	w1, h1 := a.Dims()
	w2, h2 := b.Dims()

	if w1 != w2 || h1 != h2 {
		return nil, ERROR_DIMS
	}

	res := mat.NewDense(w1, h1, nil)
	res.Sub(a, b)
	return res, nil
}

func AddScalar(scalar float64, matrix mat.Matrix) mat.Matrix {
	w, h := matrix.Dims()
	resVec := make([]float64, w*h)
	for i := range resVec {
		resVec[i] = scalar
	}

	res := mat.NewDense(w, h, resVec)
	return res
}
