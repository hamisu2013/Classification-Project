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
/**
 *
 * @author HamisuSenior
 * African University of Science and Technology Abuja 
 * 15/10/2019
 */
public class NaiveBayesClassifierModel{
    public static void main(String[]agrs)throws Exception{
        //Loading the training dataset
        DataSource source = new DataSource("Supervisedmalaria.arff");
        Instances trainDataset = source.getDataSet();
        //setting the class index to the last attribut
        trainDataset.setClassIndex(trainDataset.numAttributes()-1);
        //building the Naive Bayes model 
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(trainDataset);
        //output the NaiveBayes Model's capabilities
        System.out.println(nb.getCapabilities().toString());
       // output the model
       System.out.print(nb);
        //save the model
        weka.core.SerializationHelper.write("NaiveBayes.model",nb);
        
        //load Model 
        NaiveBayes nb2 =(NaiveBayes) weka.core.SerializationHelper.read("NaiveBayes.model");
        //Evaluating the Model
        Evaluation eval = new Evaluation(trainDataset);
        DataSource source1 = new DataSource("testData.arff");
        Instances testData = source1.getDataSet();
        //setting the class index to the last attribut
        testData.setClassIndex(testData.numAttributes()-1);
        eval.evaluateModel(nb2, testData);
        System.out.println(eval.toSummaryString("\t|||   EVALUATION RESULT  |||\n", false));
        
        //the comfusion Matrix
        System.out.println(eval.toMatrixString("\n    |=  =    OVERALL CONFUSION MATRIX  =  =|   \n"));
    
}
}