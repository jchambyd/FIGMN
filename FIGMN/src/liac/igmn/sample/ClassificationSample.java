package liac.igmn.sample;

import liac.igmn.core.IGMN;
import liac.igmn.evaluation.Evaluator;
import liac.igmn.loader.DataLoader;
import liac.igmn.loader.Dataset;
import liac.preprocessing.DatasetFilter;


public class ClassificationSample
{
	public static void main(String[] args)
	{
		try
		{
			Dataset dataset = DataLoader.loadARFF("data/iris.arff");
			DatasetFilter.normalize(dataset);
						
			IGMN igmn = new IGMN(dataset.getDataRange(), 0.1, 0.45);
			
			Evaluator evaluator = new Evaluator(true);
			evaluator.crossValidation(igmn, dataset, 10, 1, false);
			evaluator.report();			
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}
}
