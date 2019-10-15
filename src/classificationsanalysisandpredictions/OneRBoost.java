/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classificationsanalysisandpredictions;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Classifier;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.rules.OneR;
/**
 *
 * @author HamisuSenior
 */
public class OneRBoost{
    public static void main(String[]agrs)throws Exception{
    //Loading the training dataset
        DataSource source = new DataSource("Supervisedmalaria.arff");
        Instances trainDataset = source.getDataSet();
        //setting the class index to the last attribut
        trainDataset.setClassIndex(trainDataset.numAttributes()-1);
        //building the mode 
        AdaBoostM1 m1 = new AdaBoostM1();
        m1.setClassifier(new OneR());
        m1.setNumIterations(20);
        m1.buildClassifier(trainDataset);
        //output the Boosted OneR Model's capabilities
        System.out.println(m1.getCapabilities().toString());
        //output the model
        System.out.print(m1);
        //save the model
        weka.core.SerializationHelper.write("OneRBoost.model",m1);
        
        //load Model 
        AdaBoostM1 m12 =(AdaBoostM1) weka.core.SerializationHelper.read("OneRBoost.model");
        //Evaluating the Model
        Evaluation eval = new Evaluation(trainDataset);
        DataSource source1 = new DataSource("testData.arff");
        Instances testData = source1.getDataSet();
        //setting the class index to the last attribut
        testData.setClassIndex(testData.numAttributes()-1);
        eval.evaluateModel(m12, testData);
        System.out.println(eval.toSummaryString("\t|||   EVALUATION RESULT  |||\n", false));
        
        //the comfusion Matrix
        System.out.println(eval.toMatrixString("\n   | =  =    OVERALL CONFUSION MATRIX  =  =|    \n"));
    
}
}