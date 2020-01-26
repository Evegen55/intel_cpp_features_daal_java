/* file: SoftmaxLayerDenseBatch.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
 //  Content:
 //     Java example of softmax layer in the batch processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.layers.softmax.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardResultId;
import com.intel.daal.algorithms.neural_networks.layers.ForwardResultLayerDataId;
import com.intel.daal.algorithms.neural_networks.layers.ForwardInputId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardResultId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardInputId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardInputLayerDataId;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.data_management.data.HomogenTensor;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

import static com.intel.daal.examples.Util.dataRoot;

/**
 * <a name="DAAL-EXAMPLE-JAVA-SOFTMAXLAYERBATCH">
 * @example SoftmaxLayerDenseBatch.java
 */
class SoftmaxLayerDenseBatch {
    private static final String datasetFileName = dataRoot + "/data/batch/layer.csv";

    /* Softmax layer parameter */
    private static final long dimension = 1;  /* Starting data dimension index to compute softmax */

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Read datasetFileName from a file and create a tensor to store forward input data */
        Tensor data    = Service.readTensorFromCSV(context, datasetFileName);

        /* Create an algorithm to compute forward softmax layer results using default method */
        SoftmaxForwardBatch softmaxLayerForward = new SoftmaxForwardBatch(context, Float.class, SoftmaxMethod.defaultDense);

        /* Set algorithm parameters */
        softmaxLayerForward.parameter.setDimension(dimension);

        /* Set input objects for the forward softmax layer */
        softmaxLayerForward.input.set(ForwardInputId.data, data);

        /* Compute forward softmax layer results */
        SoftmaxForwardResult forwardResult = softmaxLayerForward.compute();

        /* Print the results of the forward softmax layer */
        Service.printTensor("Forward softmax layer result (first 5 rows):", forwardResult.get(ForwardResultId.value), 5, 0);

        /* Get the size of forward softmax layer output */
        int nSize = (int)forwardResult.get(ForwardResultId.value).getSize();
        long[] dims = forwardResult.get(ForwardResultId.value).getDimensions();

        /* Create a tensor with backward input data */
        double[] backData = new double[nSize];
        Tensor tensorDataBack = new HomogenTensor(context, dims, backData, 0.01);

        /* Create an algorithm to compute backward softmax layer results using default method */
        SoftmaxBackwardBatch softmaxLayerBackward = new SoftmaxBackwardBatch(context, Float.class, SoftmaxMethod.defaultDense);

        /* Set input objects for the backward softmax layer */
        softmaxLayerBackward.input.set(BackwardInputId.inputGradient, tensorDataBack);
        softmaxLayerBackward.input.set(BackwardInputLayerDataId.inputFromForward, forwardResult.get(ForwardResultLayerDataId.resultForBackward));

        /* Compute backward softmax layer results */
        SoftmaxBackwardResult backwardResult = softmaxLayerBackward.compute();

        /* Print the results of the backward softmax layer */
        Service.printTensor("Backward softmax layer result (first 5 rows):", backwardResult.get(BackwardResultId.gradient), 5, 0);

        context.dispose();
    }
}
