/* file: SmoothReLUDenseBatch.java */
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
 //     Java example of SmoothReLU algorithm
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.math;

import com.intel.daal.algorithms.math.smoothrelu.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

import static com.intel.daal.examples.Util.dataRoot;

/**
 * <a name="DAAL-EXAMPLE-JAVA-SMOOTHRELUBATCH">
 * @example SmoothReLUDenseBatch.java
 */

class SmoothReLUDenseBatch {
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
        Batch smoothReLUAlgorithm = new Batch(context, Float.class, Method.defaultDense);

        /* Set an input object for the algorithm */
        smoothReLUAlgorithm.input.set(InputId.data, input);

        /* Compute SmoothReLU function */
        Result result = smoothReLUAlgorithm.compute();

        /* Print the results of the algorithm */
        Service.printNumericTable("SmoothReLU result (first 5 rows):", result.get(ResultId.value), 5);

        context.dispose();
    }
}
