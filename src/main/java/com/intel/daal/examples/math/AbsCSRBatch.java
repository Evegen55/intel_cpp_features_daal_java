/* file: AbsCSRBatch.java */
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
 //     Java example of Abs algorithm
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.math;

import com.intel.daal.algorithms.math.abs.*;
import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

import static com.intel.daal.examples.Util.dataRoot;

/**
 * <a name="DAAL-EXAMPLE-JAVA-ABSCSRBATCH">
 * @example AbsCSRBatch.java
 */

class AbsCSRBatch {
    private static final String dataset = dataRoot + "/data/batch/covcormoments_csr.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Read a data set from a file and create a numeric table for storing the input data */
        CSRNumericTable input = Service.createSparseTable(context, dataset);

        /* Create an algorithm */
        Batch absAlgorithm = new Batch(context, Float.class, Method.fastCSR);

        /* Set an input object for the algorithm */
        absAlgorithm.input.set(InputId.data, input);

        /* Compute Abs function */
        Result result = absAlgorithm.compute();

        /* Print the results of the algorithm */
        Service.printNumericTable("Abs result (first 5 rows):", (CSRNumericTable)result.get(ResultId.value), 5);

        context.dispose();
    }
}
