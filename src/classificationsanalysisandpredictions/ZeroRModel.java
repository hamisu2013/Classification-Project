/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classificationsanalysisandpredictions;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import weka.classifiers.Evaluation;
import weka.classifiers.rules.ZeroR;
/**
 *
 * @author HamisuSenior
 * African University of Science and Technology Abuja 
 * 15/10/2019
 */
public class ZeroRModel {
    public static void main(String[]agrs)throws Exception{
    //Loading the training dataset
        DataSource source = new DataSource("Supervisedmalaria.arff");
        Instances trainDataset = source.getDataSet();
        //setting the class index to the last attribut
        trainDataset.setClassIndex(trainDataset.numAttributes()-1);
        //building the ZeroR model 
        ZeroR zeroR = new ZeroR();
        zeroR.buildClassifier(trainDataset);
        //output the ZeroR Model's capabilities
        System.out.println(zeroR.getCapabilities().toString());
        //output the model
        System.out.print(zeroR);
        //save the model
        weka.core.SerializationHelper.write("zeroR.model",zeroR);
        
        //load Model 
        ZeroR zeroR2 =(ZeroR) weka.core.SerializationHelper.read("zeroR.model");
        //Evaluating the Model
        Evaluation eval = new Evaluation(trainDataset);
        DataSource source1 = new DataSource("testData.arff");
        Instances testData = source1.getDataSet();
        //setting the class index to the last attribut
        testData.setClassIndex(testData.numAttributes()-1);
        eval.evaluateModel(zeroR2, testData);
        System.out.println(eval.toSummaryString("\t|||   EVALUATION RESULT  |||\n", false));
       
        //the comfusion Matrix
        System.out.println(eval.toMatrixString("\n    |=  =    OVERALL CONFUSION MATRIX  =  =|   \n"));
    
}
}