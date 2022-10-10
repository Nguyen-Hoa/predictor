package predictor

import "github.com/Nguyen-Hoa/worker"

type Analytical struct {
}

func (a *Analytical) Init(_ string) error {
	return nil
}

func (a *Analytical) Predict(w *worker.ManagerWorker) (float32, error) {
	min, max := w.DynamicRange[0], w.DynamicRange[1]
	util := w.LatestCPU

	prediction := ((max - min) * util * .01) + min
	return prediction, nil
}
