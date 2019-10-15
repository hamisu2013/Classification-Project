/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classificationsanalysisandpredictions;


import java.io.File;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
/**
 *
 * @author HamisuSenior
 * African University of Science and Technology Abuja 
 * 15/10/2019
 */
public class ClassificationsAnalysisAndPredictions {

     public static void main (String[] args) throws Exception {
      //load file in comma separated value File(CSV)
      CSVLoader loader = new CSVLoader();
      loader.setSource(new File("malaria.csv"));
      //get Insatances object
      Instances data = loader.getDataSet();
      
        //saving the  Attribute Relational File Format(arff)
       ArffSaver saver = new ArffSaver();
       saver.setInstances(data);
       //set the dataset we want to convert and save as arff
       saver.setFile(new File("malaria.arff"));
       saver.writeBatch();
       // printing the unsupervised dataset
       System.out.println("\n|||  THIS IS THE UNSUPERVISED ARFF AFTER CONVERSION FROM CSV |||");
       System.out.println(data);
       
       
      //Loading the  Attribute Relational File Format(arff)
      DataSource source = new DataSource("malaria.arff");
      Instances dataset = source.getDataSet();
      
      /*
      //transforming arff into supervised data suitable for classification 
      //by removing some attribute 
      String[] opts = new String[]{"-R","1"};
      //create a remove object (this is a filter class)
      Remove remove = new Remove();
      //set the filter option 
      remove.setOptions(opts);
      //pass the dataset to the filter
      remove.setInputFormat(dataset);
      //apply the filter 
      Instances newData = Filter.useFilter(dataset, remove); 
      
      
      //saving the dataset to a new file as a supervised data
       ArffSaver save = new ArffSaver();
       saver.setInstances(newData);
      //saving the supervised dataset as arff
       saver.setFile(new File("Supervisedmalaria.arff"));
       saver.writeBatch();
       */
       DataSource sources = new DataSource("Supervisedmalaria.arff");
       Instances sdataset = sources.getDataSet();
       // printing the unsupervised dataset
       System.out.println("\n|||  THIS IS THE SUPERVISED ARRF DATA AFTER PREPROCESSING   |||");
       System.out.println(sdataset);
    }
}
      
    
    

