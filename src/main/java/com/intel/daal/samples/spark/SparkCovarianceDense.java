/* file: SparkCovarianceDense.java */
/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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
//      Java sample of dense variance-covariance matrix computation in the
//      distributed processing mode
////////////////////////////////////////////////////////////////////////////////
*/

package com.intel.daal.samples.spark;

import com.intel.daal.algorithms.covariance.DistributedStep1Local;
import com.intel.daal.algorithms.covariance.DistributedStep2Master;
import com.intel.daal.algorithms.covariance.DistributedStep2MasterInputId;
import com.intel.daal.algorithms.covariance.InputId;
import com.intel.daal.algorithms.covariance.Method;
import com.intel.daal.algorithms.covariance.PartialResult;
import com.intel.daal.algorithms.covariance.Result;
import com.intel.daal.algorithms.covariance.ResultId;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.services.DaalContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;

public class SparkCovarianceDense {
    /* Class containing the algorithm results */
    static class CovarianceResult {
        public HomogenNumericTable covariance;
        public HomogenNumericTable mean;
    }

    public static CovarianceResult runCovariance(JavaRDD<HomogenNumericTable> dataRDD) {
        JavaRDD<PartialResult> partsRDD = computeStep1Local(dataRDD);
        PartialResult reducedPres = reducePartialResults(partsRDD);
        CovarianceResult result = finalizeMergeOnMasterNode(reducedPres);
        return result;
    }

    private static JavaRDD<PartialResult> computeStep1Local(JavaRDD<HomogenNumericTable> dataRDD) {
        return dataRDD.map(new Function<HomogenNumericTable, PartialResult>() {
            public PartialResult call(HomogenNumericTable table) {
                DaalContext localContext = new DaalContext();

                /* Create an algorithm to compute a dense variance-covariance matrix on local nodes*/
                DistributedStep1Local covarianceLocal = new DistributedStep1Local(localContext, Double.class, Method.defaultDense);

                /* Set the input data on local nodes */
                table.unpack(localContext);
                covarianceLocal.input.set(InputId.data, table);

                /* Compute a dense variance-covariance matrix on local nodes */
                PartialResult pres = covarianceLocal.compute();
                pres.pack();

                localContext.dispose();
                return pres;
            }
        });
    }

    private static PartialResult reducePartialResults(JavaRDD<PartialResult> partsRDD) {
        return partsRDD.reduce(new Function2<PartialResult, PartialResult, PartialResult>() {
            public PartialResult call(PartialResult pr1, PartialResult pr2) {
                DaalContext localContext = new DaalContext();

                /* Create an algorithm to compute new partial result from two partial results */
                DistributedStep2Master covarianceMaster = new DistributedStep2Master(localContext, Double.class, Method.defaultDense);

                /* Set the input data recieved from the local nodes */
                pr1.unpack(localContext);
                pr2.unpack(localContext);
                covarianceMaster.input.add(DistributedStep2MasterInputId.partialResults, pr1);
                covarianceMaster.input.add(DistributedStep2MasterInputId.partialResults, pr2);

                /* Compute a new partial result from two partial results */
                PartialResult reducedPresLocal = (PartialResult)covarianceMaster.compute();
                reducedPresLocal.pack();

                localContext.dispose();
                return reducedPresLocal;
            }
        });
    }

    private static CovarianceResult finalizeMergeOnMasterNode(PartialResult reducedPres) {

        final DaalContext context = new DaalContext();

        /* Create an algorithm to compute a dense variance-covariance matrix on the master node */
        final DistributedStep2Master covarianceMaster = new DistributedStep2Master(context, Double.class, Method.defaultDense);

        /* Set the reduced partial result to the master algorithm to compute the final result */
        reducedPres.unpack(context);
        covarianceMaster.input.add(DistributedStep2MasterInputId.partialResults, reducedPres);

        /* Compute a dense variance-covariance matrix on the master node */
        covarianceMaster.compute();

        /* Finalize computations and retrieve the results */
        final Result res = covarianceMaster.finalizeCompute();

        final CovarianceResult covResult = new CovarianceResult();
        covResult.covariance = (HomogenNumericTable)res.get(ResultId.covariance);
        covResult.mean = (HomogenNumericTable)res.get(ResultId.mean);
        return covResult;
    }
}
