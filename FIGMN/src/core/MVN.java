package core;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrices;

public class MVN implements Serializable {

    public double count = 1;
    public DenseMatrix invCov;
    public double logdet;
    public double lognormalizer;
    public DenseVector mean;
    public DenseVector diag;
    public double age = 1;
    public double[] correlateds;
    protected double chisq = Double.NaN;
    public boolean useDiagonal  = false;
    public int[] features = null;
    public double attrselection = 1;
        
    public int size() 
	{
        return this.mean.size();
    }
    
    public double getPartialCorrelationBetween(int i, int j) 
	{
        return -this.invCov.get(i, j) / Math.sqrt(this.invCov.get(i, i) * this.invCov.get(j, j));
    }

    public double getPartialCorrelationBetween(int i, int[] js) 
	{
        double sum = 0;
        for (int k = 0; k < js.length; k++) 
		{
            sum += this.getPartialCorrelationBetween(i, js[k]);
        }
        return sum / js.length;
    }
    
    public double[] getPartialCorrelations(int js[]) 
	{
        double[] c = new double[this.mean.size() - js.length];
        double total = 0;
        if (js.length == 0) 
		{
            js = Matrices.index(0, this.mean.size());
        }
        for (int i = 0; i < c.length; i++) 
		{
            c[i] = Math.abs(this.getPartialCorrelationBetween(i, js));
            total += c[i];
        }
        if (total > 0) 
		{
            for (int i = 0; i < c.length; i++)
                c[i] /= total;
        }
        return c;
    }
    
    public int[] getOrderedFeatures(int[] js, double threshold) 
	{
        Integer[] inds = new Integer[this.mean.size() - js.length];
        double total = 0;
        for (int i = 0; i < inds.length; i++) 
		{
            inds[i] = i;
        }
        double[] corrs = this.getPartialCorrelations(js);
        Arrays.sort(inds, new Comparator<Integer>() {
            @Override public int compare(final Integer o1, final Integer o2) {
                return Double.compare(corrs[o2], corrs[o1]);
            }
        });
		
        int[] f = new int[inds.length];
        int sz = inds.length;
        
		for (int i = 0; i < inds.length; i++) 
		{
            f[i] = inds[i];
            total += corrs[inds[i]];
            if (total > threshold) 
			{
                sz = i + 1;
                break;
            }
        }
        this.features = Arrays.copyOf(f, sz);
        
		return this.features;
    }
    
    public double[] getCorrelationsFor(int attr) 
	{
        if (this.correlateds != null) 
			return this.correlateds;
        int D = size();
        double[] corr = new double[D];
        DenseMatrix c = this.invCov;
        double stdAttr = c.get(attr, attr);
        if (stdAttr == 0) 
			return corr;
        double total = 0;
        for (int i = 0; i < D; i++) 
		{
            double entry = c.get(i, i);
            if (i == attr) 
				continue;
            corr[i] = Double.MIN_VALUE + Math.abs(c.get(attr, i) / (Math.sqrt(entry * stdAttr)));
            total += corr[i];
        }
        if (total == 0) 
			return corr;
        
        this.correlateds = corr;
        
		return corr;        
    }
    
    public double[] getCorrelations() 
	{
        int D = this.size();
        double[] corr = new double[D];
        double total = 0;
        for (int i = 0; i < D; i++) 
		{
            double[] corrsi = this.getCorrelationsFor(i);
            for (int j = 0; j < D; j++) 
			{
                if (i == j) 
					continue;
                corr[i] += corrsi[j];
            }
            total += corr[i];
        }
        for (int i = 0; i < D; i++)
			corr[i] /= total;
        
        return corr;
    }
    
    public int[] getMostCorrelateds(int attr, double threshold) 
	{
        int D = this.size();
        double total = 0;
        double[] corrs = this.getCorrelationsFor(attr);
        final DenseVector v = new DenseVector(corrs);
        final DenseVector v2 = v.copy().scale(1.0 / v.copy().norm(Vector.Norm.One));
        Integer[] inds = new Integer[corrs.length];
        
		for (int i = 0; i < inds.length; i++) 
            inds[i] = i;
        
		Arrays.sort(inds, new Comparator<Integer>() {
            @Override public int compare(final Integer o1, final Integer o2) {
                return Double.compare(v2.get(o2), v2.get(o1));
            }
        });
        
		int i;
        for (i = 0; i < inds.length; i++) 
		{
			total += v2.get(inds[i]);
			if (total >= threshold) 
				break;
        }        
		
        int[] r = new int[Math.min(i + 1, inds.length)];
        for (int j = 0; j < r.length; j++)
            r[j] = inds[j];
		
        return r;
    }

    public int[] getMostCorrelateds(double threshold) 
	{
        int D = this.size();
        double total = 0;
        final double[] corrs = this.getCorrelations();
        Integer[] inds = new Integer[corrs.length];
        
		for (int i = 0; i < inds.length; i++)
            inds[i] = i;
		
        Arrays.sort(inds, new Comparator<Integer>() {
            @Override public int compare(final Integer o1, final Integer o2) {
                return Double.compare(corrs[o2], corrs[o1]);
            }
        });
		
        int i;
        for (i = 0; i < inds.length; i++) 
		{
			total += corrs[inds[i]];
			if (total >= threshold) 
				break;
        }       
		
        int[] r = new int[Math.min(i+1, inds.length)];
        for (int j = 0; j < r.length; j++) 
		    r[j] = inds[j];
        
		return r;
    }
    
    public int getMostCorrelatedTo(int attr) 
	{
        double[] corrs = this.getCorrelationsFor(attr);
        int most = 0;
        double mostValue = 0;
		
        for (int i = 1; i < corrs.length; i++) 
		{
            if (corrs[i] > mostValue) 
			{
                mostValue = corrs[i];
                most = i;
            }
        }
        return most;
    }
    
    public double totalCorrelationFor(int attr) 
	{
        double mc = 0;
        int D = this.size();
        for (int i = 0; i < D; i++) 
		{
            if (i == attr) 
				continue;
            mc += this.invCov.get(attr, i);
        }
        return mc;
    }

    public MVN(double[] mean, DenseMatrix invCov, double d) throws Exception 
	{
        this.logdet = d;
        this.invCov = invCov;
        this.mean = new DenseVector(mean);
        
		if (Double.isNaN(logdet) || Double.isInfinite(this.logdet))
            throw(new Exception("Invalid log determinant: " + this.logdet));
        
		this.setLogNormalizer();
    }
    
    public void setLogNormalizer() 
	{
        this.lognormalizer = - 0.5 * (logdet +  this.mean.size() * Math.log(2*Math.PI));        
    }

    public MVN(double[] mean, DenseMatrix invCov, double d, boolean diagonal) throws Exception 
	{
        DenseVector diag = new DenseVector(new double[invCov.numColumns()]);
        for (int i = 0; i < diag.size(); i++)
            diag.set(i, invCov.get(i, i));

		if (Double.isNaN(this.logdet) || Double.isInfinite(this.logdet))
            throw(new Exception("Invalid log determinant: "+ this.logdet));
        
        this.diag = diag;
        this.useDiagonal = true;
        double[] rv = new double[mean.length];
        
		for (int i = 0; i < mean.length; i++)
            rv[i] = 1 / diag.get(i);
        
        this.diag = new DenseVector(rv);
        this.mean = new DenseVector(mean);
        this.setLogNormalizer();
        System.out.println("Using diagonal covariance");
    }
    
    public MVN(double[] mean, DenseVector diag, double d) throws Exception 
	{
        if (Double.isNaN(this.logdet) || Double.isInfinite(this.logdet))
            throw(new Exception("Invalid log determinant: " + this.logdet));
                
        this.diag = diag;
        this.useDiagonal = true;
        double[] rv = new double[mean.length];
		
        for (int i = 0; i < mean.length; i++)
			rv[i] = 1 / diag.get(i);
		
        this.diag = new DenseVector(rv);
        this.mean = new DenseVector(mean);
        
		this.setLogNormalizer();        
    }
    
    public double[] getMeans() 
	{
        return this.mean.getData();
    }

    @Override
    public String toString() 
	{
        return this.count + ":" + Arrays.toString(this.getMeans()) + "\n";
    }

    public DenseVector getMeansVector() 
	{
        return this.mean.copy();
    }
    
    public double mahalanobis(final DenseVector x) 
	{
        ArrayList<Integer> inds = new ArrayList<>();
        ArrayList<Integer> outputs = new ArrayList<>();
        
        for (int i = 0; i < x.size(); i++) 
		{
            if (Double.isNaN(x.get(i))) 
			{
               x.set(i, this.getMeansVector().get(i));
               outputs.add(i);
            }
            else 
			    inds.add(i);
        }
		
        int[] inds_ = inds.stream().mapToInt(i -> i).toArray();
        int[] outputs_ = outputs.stream().mapToInt(i -> i).toArray();
        DenseVector diff = (DenseVector) x.add(-1, mean);
        
		if (this.useDiagonal || this.count == 1) 
			return this.diagMahalanobis(diff);
		
        if (this.attrselection < 1) 
		    inds_ = this.getOrderedFeatures(outputs_, attrselection);            
        
		DenseVector subdiff = new DenseVector(Matrices.getSubVector(diff, inds_));
        DenseVector temp = new DenseVector(subdiff.size());
        DenseMatrix subinvcov = new DenseMatrix(Matrices.getSubMatrix(this.invCov, inds_, inds_));
        subinvcov.mult(subdiff, temp);
		
        return temp.dot(subdiff);            
	}
    
    public double diagMahalanobis(DenseVector diff) 
	{
        double dist = 0;
        
		if (this.diag == null) 
		{
            this.diag = new DenseVector(this.invCov.numColumns());
            
			for (int i = 0; i < this.invCov.numRows(); i++)
				this.diag.set(i, this.invCov.get(i, i));
        }
		
        for (int i = 0; i < diff.size(); i++) 
            dist += Math.pow(diff.get(i), 2) * this.diag.get(i);
        
		return dist;
    }
    
    public double diagMahalanobis(DenseVector diff, double max) 
	{
        double dist = 0;
        for (int i = 0; i < diff.size(); i++) 
		{
            dist += Math.pow(diff.get(i), 2) / this.diag.get(i);
            if (dist > max) 
				break;
        }
        return dist;
    }

    public double mahalanobis(final double[] x) 
	{
        return this.mahalanobis(new DenseVector(x));
    }
       
    public DenseMatrix getInverseCov() 
	{
        return this.invCov;
    }
   
    public static int countNaNs(double[] input) 
	{
        int c = 0;
        
		for (int i = 0; i < input.length; i++) 
		{
            if (Double.isNaN(input[i]))
                c++;
        }		
        return c;
    }

    public double density(final double[] input) 
	{
        return Math.exp(this.logdensity(input));
    }

    public double logdensity(final double[] input) 
	{
        double[] ret = input.clone();
		
        for (int i = 0; i < input.length; i++) 
		{
            if (Double.isNaN(input[i]))
                ret[i] = this.getMeans()[i];            
        }
        double d = -0.5 * this.mahalanobis(ret) + this.lognormalizer;

		return d;
    }

    private void readObject(
      ObjectInputStream aInputStream
    ) throws ClassNotFoundException, IOException {
        aInputStream.defaultReadObject();
    }
   
    private void writeObject(
      ObjectOutputStream aOutputStream
    ) throws IOException {
      aOutputStream.defaultWriteObject();
    }   
}
