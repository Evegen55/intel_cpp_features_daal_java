/* file: SampleImplicitALSCSR.java */
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
 //     Java sample of the implicit alternating least squares (ALS) algorithm.
 //
 //     The program trains the implicit ALS trainedModel on a supplied training data
 //     set.
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.samples.spark;

import com.intel.daal.algorithms.implicit_als.prediction.ratings.RatingsResult;
import com.intel.daal.algorithms.implicit_als.prediction.ratings.RatingsResultId;
import com.intel.daal.algorithms.implicit_als.training.DistributedPartialResultStep4;
import com.intel.daal.algorithms.implicit_als.training.DistributedPartialResultStep4Id;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;
import scala.Tuple3;

import java.io.IOException;
import java.nio.DoubleBuffer;
import java.util.List;

import static com.intel.daal.examples.Util.dataRoot;

public class SampleImplicitALSCSR {

    public static void main(String[] args) throws IOException, ClassNotFoundException, IllegalAccessException, InterruptedException {

        /* Create JavaSparkContext that holds SparkContext, loads defaults from the system properties and the classpath and sets the name */
        final SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("Spark Implicit ALS");
        final JavaSparkContext javaSparkContext = new JavaSparkContext(sparkConf);

        /* Read from the distributed HDFS data set at a specified path */
        final DistributedHDFSDataSet ddTrain = new DistributedHDFSDataSet(dataRoot + "/data/spark/ImplicitALSCSRTrans", javaSparkContext);

        final JavaPairRDD<Integer, NumericTable> dataRDD = ddTrain.getCSRAsPairRDDWithIndex();

        final SparkImplicitALSCSR.TrainingResult trainedModel = SparkImplicitALSCSR.trainModel(javaSparkContext, dataRDD);
        printTrainedModel(trainedModel);

        final JavaRDD<Tuple3<Integer, Integer, RatingsResult>> predictedRatings =
            SparkImplicitALSCSR.testModel(trainedModel.usersFactors, trainedModel.itemsFactors);

        printPredictedRatings(predictedRatings);

//        Thread.sleep(Long.MAX_VALUE);

        javaSparkContext.stop();
    }

    public static void printTrainedModel(SparkImplicitALSCSR.TrainingResult trainedModel) throws IllegalAccessException {
        DaalContext context = new DaalContext();
        List<Tuple2<Integer, DistributedPartialResultStep4>> itemsFactorsList = trainedModel.itemsFactors.collect();
        for (Tuple2<Integer, DistributedPartialResultStep4> tup : itemsFactorsList) {
            tup._2.unpack(context);
            printNumericTable("Partial items factors " + tup._1 + " :",
                              tup._2.get(DistributedPartialResultStep4Id.outputOfStep4ForStep1).getFactors());
            tup._2.pack();
        }

        List<Tuple2<Integer, DistributedPartialResultStep4>> usersFactorsList = trainedModel.usersFactors.collect();
        for (Tuple2<Integer, DistributedPartialResultStep4> tup : usersFactorsList) {
            tup._2.unpack(context);
            printNumericTable("Partial users factors " + tup._1 + " :",
                              tup._2.get(DistributedPartialResultStep4Id.outputOfStep4ForStep1).getFactors());
            tup._2.pack();
        }
        context.dispose();
    }

    public static void printPredictedRatings(JavaRDD<Tuple3<Integer, Integer, RatingsResult>> predictedRatings) throws IllegalAccessException {
        DaalContext context = new DaalContext();
        List<Tuple3<Integer, Integer, RatingsResult>> predictedRatingsList = predictedRatings.collect();
        for (Tuple3<Integer, Integer, RatingsResult> tup : predictedRatingsList) {
            tup._3().unpack(context);
            printNumericTable("Ratings [" + tup._1() + ", " + tup._2() + "]" , tup._3().get(RatingsResultId.prediction));
            tup._3().pack();
        }
        context.dispose();
    }

    public static void printNumericTable(String header, NumericTable nt, long nPrintedRows, long nPrintedCols) throws IllegalAccessException {
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

    public static void printNumericTable(String header, NumericTable nt, long nRows) throws IllegalAccessException {
        printNumericTable(header, nt, nRows, nt.getNumberOfColumns());
    }

    public static void printNumericTable(String header, NumericTable nt) throws IllegalAccessException {
        printNumericTable(header, nt, nt.getNumberOfRows());
    }
}
