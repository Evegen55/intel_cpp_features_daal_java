/* file: TanhDenseBatch.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
 //  Content:
 //     Java example of Hyperbolic tangent algorithm
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.math;

import com.intel.daal.algorithms.math.tanh.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

import static com.intel.daal.examples.Util.dataRoot;

/**
 * <a name="DAAL-EXAMPLE-JAVA-TANHDENSEBATCH">
 * @example TanhDenseBatch.java
 */

class TanhDenseBatch {
    private static final String dataset = dataRoot + "/data/batch/covcormoments_dense.csv";
    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Retrieve the input data */
        FileDataSource dataSource = new FileDataSource(context, dataset,
                                                       DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                                                       DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();

        NumericTable input = dataSource.getNumericTable();

        /* Create an algorithm */
        Batch tanhAlgorithm = new Batch(context, Float.class, Method.defaultDense);

        /* Set an input object for the algorithm */
        tanhAlgorithm.input.set(InputId.data, input);

        /* Compute Hyperbolic tangent function */
        Result result = tanhAlgorithm.compute();

        /* Print the results of the algorithm */
        Service.printNumericTable("Tanh result (first 5 rows):", result.get(ResultId.value), 5);

        context.dispose();
    }
}
