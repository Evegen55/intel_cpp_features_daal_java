/* file: SampleCovarianceCSR.java */
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
 //     Java sample of sparse variance-covariance matrix computation
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.samples.spark;

import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.StringDataSource;
import com.intel.daal.services.DaalContext;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.IOException;
import java.nio.DoubleBuffer;

import static com.intel.daal.examples.Util.dataRoot;

public class SampleCovarianceCSR {

    public static void main(String[] args) throws IOException, InterruptedException {

        final DaalContext daalContext = new DaalContext();

        /* Create JavaSparkContext that loads defaults from the system properties and the classpath and sets the name */
        final SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("Spark covariance(CSR)");
        final JavaSparkContext javaSparkContext = new JavaSparkContext(sparkConf);

        /* Read from the distributed HDFS data set at a specified path */
        StringDataSource templateDataSource = new StringDataSource(daalContext, "");
        DistributedHDFSDataSet dd = new DistributedHDFSDataSet(dataRoot + "/data/spark/CovarianceCSR/", templateDataSource);
        JavaRDD<CSRNumericTable> dataRDD = dd.getCSRAsRDD(javaSparkContext);

        /* Compute a sparse variance-covariance matrix for dataRDD */
        SparkCovarianceCSR.CovarianceResult result = SparkCovarianceCSR.runCovariance(daalContext, dataRDD);

        /* Print the results */
        HomogenNumericTable Covariance = result.covariance;
        HomogenNumericTable Mean = result.mean;

        printNumericTable("Covariance matrix (upper left square 10*10) :", Covariance, 10, 10);
        printNumericTable("Mean vector:", Mean, 1, 10);

        daalContext.dispose();
//        Thread.sleep(Long.MAX_VALUE); //if you want to see ho Spark actually does a job
        javaSparkContext.stop();
    }

    public static void printNumericTable(String header, NumericTable nt, long nPrintedRows, long nPrintedCols) {
        long nNtCols = nt.getNumberOfColumns();
        long nNtRows = nt.getNumberOfRows();
        long nRows = nNtRows;
        long nCols = nNtCols;

        if (nPrintedRows > 0) {
            nRows = Math.min(nNtRows, nPrintedRows);
        }

        DoubleBuffer result = DoubleBuffer.allocate((int) (nNtCols * nRows));
        result = nt.getBlockOfRows(0, nRows, result);
        if (nPrintedCols > 0) {
            nCols = Math.min(nNtCols, nPrintedCols);
        }

        StringBuilder builder = new StringBuilder();
        builder.append(header);
        builder.append("\n");
        for (long i = 0; i < nRows; i++) {
            for (long j = 0; j < nCols; j++) {
                String tmp = String.format("%-6.3f   ", result.get((int) (i * nNtCols + j)));
                builder.append(tmp);
            }
            builder.append("\n");
        }
        System.out.println(builder.toString());
    }
}
