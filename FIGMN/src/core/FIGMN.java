package core;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.logging.Level;
import java.util.logging.Logger;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrices;
import no.uib.cipr.matrix.Vector;
import weka.classifiers.AbstractClassifier;
import weka.core.AdditionalMeasureProducer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class FIGMN extends AbstractClassifier implements Serializable, AdditionalMeasureProducer {
    
    protected ArrayList<MVN> distributions;
    private double[][] sigmaIni;
    private double logDetSigmaIni;
    private boolean verbose = true;
    private boolean skipComponents = true;
    private boolean autoSigma = false;
    private double attrSelection = 1;
    protected double beta = Double.MIN_VALUE;
    protected double delta = 1;//0.3;
    private double prune = Double.POSITIVE_INFINITY;
    protected double chisq = Double.NaN;
    private int inputLength = -1;
    static final long serialVersionUID = 3932117032546553728L;
    int cores;
    public boolean useDiagonal = false;
    
    public FIGMN() 
	{
        this.cores = Runtime.getRuntime().availableProcessors();
        this.distributions = new ArrayList<>();
    }
    
    public FIGMN(double[][] data, double delta) 
	{
        this();
        this.delta = delta;
        this.inputLength = data[0].length;
        this.updateSigmaIni(data.clone());
    }
    
    public FIGMN(double delta, double[][] sigmaIni) 
	{
        this();
        this.delta = delta;
        this.sigmaIni = sigmaIni.clone();
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

    private void updateSigmaIni(double[] datum) 
	{
        int D = datum.length;
        this.sigmaIni = Matrices.getArray(Matrices.identity(D).scale(delta));
        this.logDetSigmaIni = 0;
        for (int i = 0; i < this.sigmaIni.length; i++) 
		{
            if (Double.isNaN(this.sigmaIni[i][i]) || this.sigmaIni[i][i] < Float.MIN_VALUE)
                this.sigmaIni[i][i] = Float.MIN_VALUE;
            this.logDetSigmaIni += Math.log(this.sigmaIni[i][i]);
            this.sigmaIni[i][i] = 1.0 / this.sigmaIni[i][i];
        }
    }
    
    private void updateSigmaIni(double[][] data) 
	{
        int N = data.length;
        if (N == 0) 
			return;
        int D = data[0].length;
        double[] mean = new double[D];
        double[] counts = new double[D];
        for (int i = 0; i < N; i++) 
		{
            for (int j = 0; j < D; j++) 
			{
                if (!Double.isNaN(data[i][j])) 
				{
                    mean[j] += data[i][j];
                    counts[j]++;
                }
            }
        }
		
        for (int j = 0; j < D; j++) 
		{
            mean[j] /= counts[j];
        }
		
        double[] var = new double[D];
        for (int i = 0; i < N; i++) 
		{
            for (int j = 0; j < D; j++) 
			{
                if (!Double.isNaN(data[i][j])) 
				{
                    var[j] += Math.pow(data[i][j] - mean[j],2) / (counts[j] - 1);
                }
            }
        }
        for (int i = 0; i < var.length; i++) 
		{			
            if (var[i] < Double.MIN_VALUE)
                var[i] = max(var);
        }
        this.sigmaIni = Matrices.getArray(FIGMN.diagonalMatrix(new DenseVector(var)).scale(delta));         
        this.logDetSigmaIni = 0;
        for (int i = 0; i < sigmaIni.length; i++) 
		{
            if (Double.isNaN(sigmaIni[i][i]) || sigmaIni[i][i] < Float.MIN_VALUE)
                this.sigmaIni[i][i] = Float.MIN_VALUE;
            this.logDetSigmaIni += Math.log(sigmaIni[i][i]);
            this.sigmaIni[i][i] = 1.0 / sigmaIni[i][i];
        }
        output("SigmaIni: " + Arrays.deepToString(sigmaIni));
    }

    public void output(String text) 
	{
        if (verbose) {
            System.out.println(text);
        }
    }
    
    public String deltaTipText() 
	{
        return "Initial component size as a factor (0 and above) of the dataset standard deviations for each dimension.";
    }

    public void setDelta(double value) 
	{
        this.delta = value;
    }
    
    public double getDelta() 
	{
        return this.delta;
    }
    
    public double getPrune() 
	{
        return this.prune;
    }
    
    public void setPrune(double value) 
	{
        this.prune = value;
    }

    public boolean getSkipComponents() 
	{
        return this.skipComponents;
    }
    
    public void setSkipComponents(boolean value) 
	{
        this.skipComponents = value;
    }

    public boolean getAutoSigma() 
	{
        return this.autoSigma;
    }
    
    public void setAutoSigma(boolean value) 
	{
        this.autoSigma = value;
    }

    public double getAttrSelection() 
	{
        return this.attrSelection;
    }
    
    public void setAttrSelection(double value)
	{
        this.attrSelection = value;
    }

    public String betaTipText() 
	{
        return "Minimum activation (between 0 and 1) for updating component.";
    }
    
    public void setBeta(double value) 
	{
        this.beta = value;
        this.chisq = Statistics.chi2inv(1 - beta, this.inputLength);
    }
    
    public double getBeta() 
	{
        return this.beta;
    }

    public String verboseTipText() 
	{
        return "Whether to log each processing step or not.";
    }
    
    public void setVerbose(boolean value) 
	{
        this.verbose = value;
    }
    
    public boolean getVerbose() 
	{
        return this.verbose;
    }
    
    public ArrayList<MVN> getDistributions() 
	{
        return distributions;
    }
    
    public void setSigmaIni(double[][] value) 
	{
        sigmaIni = value.clone();
    }
    
    public void setSigmaIni(double[] value) 
	{
        sigmaIni = Matrices.getArray(FIGMN.diagonalMatrix(new DenseVector(value)).scale(delta));
    }
    
    public void setSigmaIni(DenseVector value) 
	{
        setSigmaIni(value.getData());
    }    

    public double[] getLikelihoods(final double[] x) 
	{
        double[] activations = this.distributions.stream().mapToDouble(mvn -> mvn.logdensity(x)).toArray();
        final double max = Arrays.stream(activations).max().getAsDouble();
        activations = Arrays.stream(activations).map(a -> a - max).toArray();
        return activations;
    }

    public double[] getDistances(final double[] x) 
	{
        double[] activations = this.distributions.stream().parallel().mapToDouble(mvn -> mvn.mahalanobis(x)).toArray();
        return activations;
    }
    
    public double[] getPosteriors(double[] x) 
	{
        int s = distributions.size();
        if (s == 1) 
			return new double[]{1};
        double[] activations = this.getLikelihoods(x.clone());
        double[] posteriors = new double[s];
        double total = 0;
        
		for (int i = 0; i < s; i++) 
		{
            posteriors[i] = Math.exp(activations[i] + Math.log(getPrior(i)));
            total += posteriors[i];
        }
        for (int i = 0; i < s; i++) 
		{
            posteriors[i] = posteriors[i] / total;
        }
		
        return posteriors;
    }

    public static double min(double[] x) 
	{
        double m = Double.POSITIVE_INFINITY;
        for (double m_ : x) 
		{
            if (m_ < m)
                m = m_;
        }
        return m;
    }
    
    public static double max(double[] x) 
	{
        double m = Double.NEGATIVE_INFINITY;
        for (double m_ : x) 
		{
            if (m_ > m)
                m = m_;
        }
        return m;
    }
    
    public static DenseMatrix diagonalMatrix(Vector v) 
	{
        DenseMatrix m = new DenseMatrix(v.size(), v.size());
		
        for (int i = 0; i < v.size(); i++) 
		{
            m.set(i, i, v.get(i));
        }
        return m;
    }    

    protected boolean testUpdate(double[] input) 
	{
        return testUpdate(input, 0);
    }

    protected boolean testUpdate(double[] input, int numClasses) 
	{
        if (Double.isNaN(this.chisq)) 
		{
            if (this.beta == 0)
                this.chisq = Double.POSITIVE_INFINITY;
            else
                this.chisq = Statistics.chi2inv(1 - this.beta, input.length);
        }
        int s = distributions.size();
        for (int i = 0; i < s; i++) 
		{
            if (distributions.get(i).mahalanobis(input.clone()) < this.chisq) 
				return true;
        }
        return false;        
    }

    public void learn(final double[] input) throws Exception 
	{
        learn(input, 0);
    }
    
    public void learn(final double[] input, int numClasses) throws Exception 
	{
        int s = distributions.size();
        if (s > 0 && this.beta < 1 && (this.beta == 0 || testUpdate(input.clone(), numClasses))) 
		{
            output(totalCount() + " Updating...");
            updateComponents(input.clone());
        }
        else 
		{
            output("Adding... " + (s + 1));
            addComponent(input.clone());
        }
        if (!Double.isInfinite(this.prune)) 
		{
            output("Pruning...");
            pruneComponents();
        }
    }
    
    public void learn(double[][] inputs) throws Exception 
	{
        for (int i = 0; i < inputs.length; i++) 
		{
            output("Learning datum #" + i);
            learn(inputs[i]);
        }
    }

    public void updateComponents(double[] x) 
	{
        int s = this.distributions.size();
        double meanP = 1.0 / s;
        double[] activations = getPosteriors(x.clone());
        double highP = max(activations);
        
        for (int i = s - 1; i >= 0; i--) 
		{
            if (!this.skipComponents || activations[i] >= meanP || activations[i] >= highP) {
                updateComponent(i, x, 1);
                break;
            }
        }
    }
    
    public static DenseVector subVector(DenseVector v, int[] indices) 
	{
        return (DenseVector) Matrices.getSubVector(v, indices);        
    }
    
    public void updateComponent(int i, final double[] x, final double activation) 
	{
        MVN mvn = this.distributions.get(i);
        DenseVector mean = mvn.mean;
        int D = mean.size();
        DenseVector input = new DenseVector(x);
        mvn.count += activation;
        double learningRate = activation / mvn.count;
        DenseVector diff = (DenseVector) input.copy().add(-1, mean);
        diff.scale(learningRate);
        
        //New mean
        mean.add(diff);
        input.add(-1, mean);
        DenseMatrix invCov = new DenseMatrix(mvn.invCov);
        DenseVector newdiag = null;
        double logdet;
        
		invCov.scale(1.0 / (1 - learningRate));
		input.scale(Math.sqrt(learningRate));
		logdet = rankOneMatrixUpdate(invCov, input, D * Math.log1p(-learningRate) + mvn.logdet);
		logdet = rankOneMatrixDowndate(invCov,diff, logdet);
		
        try 
		{ 
			if (Double.isInfinite(logdet)) 
				logdet = Double.MAX_VALUE;
			mvn.invCov = invCov;
			mvn.age++;
			mvn.features = null;
			
			if (Double.isNaN(logdet) || Double.isInfinite(logdet)) 
				throw(new Exception("Invalid log determinant: "+logdet));
			
			mvn.logdet = logdet;
			mvn.setLogNormalizer();
        }
        catch (Exception e) {
            System.out.println(e);
            if (verbose) 
			    Logger.getLogger(FIGMN.class.getName()).log(Level.WARNING, "Precision Matrix Error: {0}", new Object[]{e.getMessage()});            
        }
    }
    
    public static DenseMatrix outerProduct(DenseVector v1, DenseVector v2) 
	{
        DenseMatrix m = new DenseMatrix(v1.size(), v1.size());
        for (int i = 0; i < v1.size(); i++) 
		{
            for (int j = 0; j < v2.size(); j++) 
			{
                m.set(i, j, v1.get(i) * v2.get(j));                
            }            
        }
        return m;
    }
    
    public double rankOneMatrixUpdate(DenseMatrix m, DenseVector v, double det) 
	{
        DenseVector temp1 = new DenseVector(v.size());
        m.mult(v, temp1);
        double temp2 = 1 + temp1.dot(v);
        double newdet = det;
        if (temp2 > 0) 
            newdet += Math.log(temp2);        
        m.add(-1, FIGMN.outerProduct(temp1, temp1).scale(1.0 / temp2));        
        
		return newdet;
    }
    
    public double rankOneMatrixDowndate(DenseMatrix m, final DenseVector v, double det) 
	{
        DenseVector temp1 = new DenseVector(v.size());
        m.mult(v, temp1);
        double temp2 = 1 - temp1.dot(v);
        double newdet = det;
        if (temp2 > 0) 
            newdet += Math.log(temp2);
        m.add(FIGMN.outerProduct(temp1,temp1).scale(1.0 / temp2));
		
        return newdet;
    }

    public int getNumComponents() 
	{
        return distributions.size();
    }
    
    protected void clearComponents() 
	{
        distributions.clear();
    }
    
    public double totalCount() 
	{
        double c = 0;
        for (MVN distribution : this.distributions) {
            c += distribution.count;
        }
        return c;
    }
    
    public double totalAge() 
	{
        double c = 0;
        for (MVN distribution : this.distributions) {
            c += distribution.age;
        }
        return c;
    }

    public double meanAge() 
	{
        return this.totalAge() / this.distributions.size();
    }
 
    public double getPrior(int i) 
	{
        return this.distributions.get(i).count / this.totalCount();
    }
    
    public double meanCount() 
	{
        return this.totalCount() / this.distributions.size();
    }
    
    public double varCount() 
	{
        int d = 0;
        int K = this.distributions.size();
        double m = this.meanCount();
        for (int i = 0; i < K; i++) 
		{
            d += Math.pow(this.distributions.get(i).count - m, 2);
        }
        return d / K;
    }
    
    public double stdCount() 
	{
        return Math.sqrt(this.varCount());
    }

    public void pruneComponents() 
	{
        if (Double.isInfinite(prune)) 
			return;
        int K = this.distributions.size();
        double m = this.meanCount();
        double am = this.meanAge();
        
        for (int i = K-1; i >= 0; i--) 
		{
            MVN mvn = distributions.get(i);
            double c = mvn.count;
            double a = mvn.age;
            if (c < m && a > am) 
			{
                verbose = true;
                output("Pruning component #" + i+" with counter "+c+" and age "+a+" (mean count: "+m+" mean age: "+am+")");
                verbose = false;
                distributions.remove(i);
            }
        }
    }

    /** recall
     * 
     * @param input
     * @return Input reconstruction.
     * @throws Exception 
     */
    public double[] recall(double[] input) throws Exception 
	{
        int s = this.distributions.size();
        double[] p = this.getPosteriors(input.clone());
        double meanP = 1.0 / s;
        double highP = max(p);
        DenseVector r = new DenseVector(new double[input.length]);
        double total = 0;
        for (int i = 0; i < s; i++) 
		{
            if (!skipComponents || p[i] >= meanP || p[i] >= highP) 
			{
                r.add(p[i], new DenseVector(recallFor(i,input)));
                total += p[i];
            }            
        }
        return r.scale(1/total).getData();
    }
    
    public static Vector subVector(final Vector v, int[] indices) 
	{
        return Matrices.getSubVector(v, indices);    
    }
    
    public static Vector subVector(final Vector v, int start, int len) 
	{
        return Matrices.getSubVector(v, Matrices.index(start, start + len));
    }
        
    public static Vector setSubVector(final Vector v, int start, final Vector v2) 
	{
        Vector vnew = v;
        for (int i = 0; i < v2.size(); i++) 
		{
            vnew.set(start + i, v2.get(i));
        }
        return vnew;
    }
    
    protected double[] recallFor(int i, final double[] input) 
	{
        int l = input.length;
        double[] inp = input.clone();
        MVN mvn = distributions.get(i);
        int noutputs = 0;
        for (int j = 0; j < l; j++) 
		{
            if (Double.isNaN(inp[j])) 
			{
                inp[j] = mvn.getMeans()[j];
                noutputs++;
            }
        }
        int[] outputs = new int[noutputs];
        int k = 0;
        for (int j = 0; j < l; j++) 
		{
            if (Double.isNaN(input[j])) 
			{
                outputs[k] = j;
                k++;
            }
        }

        DenseVector m = mvn.getMeansVector();
        if (useDiagonal || mvn.count == 1) 
		{
            return m.getData();
        }
        DenseMatrix ci = new DenseMatrix(mvn.invCov);
        DenseVector a = new DenseVector(inp);
        int aStart = 0, aLength = l-noutputs, aEnd = l-1-noutputs;
        int bStart = l-noutputs, bLength = noutputs, bEnd = l-1;
        DenseVector ma;
        DenseMatrix cba;
        if (this.attrSelection < 1) 
		{
            mvn.features = null;
            int[] r = mvn.getOrderedFeatures(outputs, this.attrSelection);
            System.out.println("Feature Selection: "+Arrays.toString(r)+" for "+Arrays.toString(outputs)+" Total: "+r.length);            
            a = new DenseVector(Matrices.getSubVector(a, r));
            ma = new DenseVector (Matrices.getSubVector(m, r));
            cba = new DenseMatrix(Matrices.getSubMatrix(ci, Matrices.index(bStart, bStart+bLength), r));
        }
        else 
		{
            a = new DenseVector (FIGMN.subVector(a, aStart, aLength));
            ma = new DenseVector (FIGMN.subVector(m, aStart, aLength));
            cba = new DenseMatrix (Matrices.getSubMatrix(ci, Matrices.index(bStart, bStart + bLength), Matrices.index(aStart, aStart + aLength)));
        }

        DenseVector mb = new DenseVector(Matrices.getSubVector(m, Matrices.index(bStart, bStart+bLength)));
        a.add(-1,ma);
        DenseMatrix cb = new DenseMatrix(Matrices.getSubMatrix(ci, Matrices.index(bStart, bEnd+1), Matrices.index(bStart, bEnd+1)));
        DenseMatrix I = (DenseMatrix)(Matrices.identity(cb.numColumns()));
        DenseMatrix cbinv = I;
        cb.solve(I, cbinv);
        DenseMatrix cbinvcba = new DenseMatrix(mb.size(),a.size());
		cbinv.mult(-1, cba, cbinvcba);

        DenseVector temp = new DenseVector(bLength);
        cbinvcba.mult(a, temp);
        mb.add(temp);        
        m = new DenseVector(setSubVector(m, bStart, mb));
        return m.getData();
    }

    public double[] nearestDistance(double[] x) 
	{
        double ndist = Double.MAX_VALUE, dist;
        double[] ndists = new double[x.length], dists = new double[x.length];
        for (MVN distribution : this.distributions) 
		{
            dist = 0;
            for (int j = 0; j < distribution.getMeans().length; j++) 
			{
                dists[j] = Math.pow(distribution.getMeans()[j] - x[j], 2);
                
				if (dists[j] <= sigmaIni[j][j]) 
					dists[j] = sigmaIni[j][j];            
                
				dist += dists[j];
                
				if (dist >= ndist) 
					break;
            }
            if (dist < ndist) 
			{
                ndist = dist;
                ndists = dists.clone();
            }
        }
        return ndists;
    }
    
    public double[][] calcSigmaIni(double[] x) 
	{
        int s = this.distributions.size();
        double invP = 1.0 / s;
        int l = x.length;
        double totalCount = 0;
        if (s > 0) 
		{
            DenseMatrix sig = new DenseMatrix(l,l);
            for (int i = 0; i < s; i++) 
			{
                MVN mvn = this.distributions.get(i);
                double c = mvn.count;
                if (c < invP) 
					continue;
                sig.add(c, mvn.invCov);
                totalCount += c;
            }
            sig.scale(1.0 / totalCount);
            
			for (int i = 0; i < l; i++) 
			{
                sigmaIni[i][i] = sig.get(i, i);
            }
        }
		
        return sigmaIni;
    }

    public double[][] calcSigmaIniDiag(double[] x) 
	{
        int s = this.distributions.size();
        double invP = 1.0 / s;
        int l = x.length;
        double totalCount = 0;
        
		if (s > 0) 
		{
            DenseVector sig = new DenseVector(new double[l]);
            for (int i = 0; i < s; i++) 
			{
                MVN mvn = this.distributions.get(i);
                double c = mvn.count;
                if (c < invP) 
					continue;
                sig.add(c, mvn.diag);
                totalCount += c;
            }
            sig.scale(1.0 / totalCount);
            
			for (int i = 0; i < l; i++) 
			{
                sigmaIni[i][i] = sig.get(i);
            }            
        }
		
        return sigmaIni;
    }
    
    public double calcDetSigmaIni(double[][] si) 
	{
        double d = 0;
        double log1 = Math.log(1.0);
        for (int i = 0; i < si.length; i++) 
		{
            d += (log1 - Math.log(si[i][i]));
        }
        return d;
    }
    
    public void addComponent(double[] mean) throws Exception 
	{
        MVN newComp;
        if (this.autoSigma) 
		{
            double[][] si = calcSigmaIni(mean.clone());
            newComp = new MVN(mean, new DenseMatrix(mean.length, mean.length), calcDetSigmaIni(si));
        }
        else 
		{
            if (useDiagonal) 
			{
                newComp = new MVN(mean, new DenseMatrix(sigmaIni), logDetSigmaIni, true);            
            }
            else 
			{                
                newComp = new MVN(mean, new DenseMatrix(sigmaIni), logDetSigmaIni);
            }
        }
        newComp.attrselection = this.attrSelection;
        distributions.add(newComp);
    }
    
    public static double[][] instancesToDoubleArrays(Instances data) 
	{
        int D = data.numAttributes(), N = data.size(), C = data.numClasses();
        double [][] result = new double[N][D];
		
        for (int i = 0; i < N; i++) 
		{
            result[i] = instanceToArray(data.get(i));
        }
        return result;
    }
    
	public static double[] instanceToArray(Instance ins) 
	{
        double[] input;
		
        if (ins.classAttribute().isNominal()) 
		{
            input = new double[ins.numAttributes() + ins.numClasses() - 1];
            for (int i = 0; i < ins.numAttributes() - 1; i++) 
			     input[i] = ins.value(i);
			
            for (int i = 0; i < ins.numClasses(); i++) 
			{
                if (ins.classIsMissing()) 
					input[i + ins.numAttributes() - 1] = Double.NaN;
                else if (ins.classValue() == i) 
					input[i + ins.numAttributes() - 1] = 1;                    
                else 
                    input[i + ins.numAttributes() - 1] = 0;
            }
        }
        else 
            input = ins.toDoubleArray();
        
		return input;
    }
	

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) 
	{
        DataSource source;
        try 
		{
            source = new DataSource("data/iris.arff");
            Instances dt = source.getDataSet();
            //dt.randomize(new Random());
            if (dt.classIndex() == -1)
               dt.setClassIndex(dt.numAttributes() - 1);
            double[][] data;
            data = FIGMN.instancesToDoubleArrays(dt);
            FIGMN igmn = new FIGMN(data,0.1);
            igmn.verbose = false;
            igmn.learn(data);
            System.out.println("# of Components: ["+igmn.distributions.size() + "] List: " + igmn.distributions);
            int c = dt.classIndex();
            double err = 0;
            int errcount = 0;
            for (int i = 0; i < data.length; i++) 
			{
                double[] input = data[i].clone();
                input[c] = Double.NaN;
                double[] result = igmn.recall(input);
                double err_ = Math.pow(result[c] - data[i][c],2);
                System.out.println("#"+i+" Target: " + data[i][c] + " Output: " + result[c] + " Error: " + err_ + " Reconstruction: " + Arrays.toString(result));
                err += err_;
                if (err_ != 0) errcount++;
            }
            err /= data.length;
            System.out.println("MSE: " + err + " Errors: " + errcount + " Accuracy: " + (data.length - errcount)/(double)data.length);
        } 
		catch (Exception ex) 
		{
            //Logger.getLogger(FIGMN.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    
    public int numberOfClusters() 
	{
        return this.distributions.size();
    }
        
    public int clusterInstance(double[] ins) throws Exception 
	{
        double dens;
        double maxDens = 0;
        int best = -1;
        for (int i = 0; i < this.numberOfClusters(); i++) 
		{
            dens = this.distributions.get(i).density(ins);
            if (dens > maxDens) 
			{
                maxDens = dens;
                best = i;
            }
        }
		
        return best;
    }

    public int clusterInstance(Instance ins) throws Exception {
        return clusterInstance(ins.toDoubleArray());
    }
    
    public int clusterInstance(Double[] ins) throws Exception {
        return clusterInstance(FIGMN.DoubleArrayToPrimive(ins));
    }

    public double[] distributionForInstance(double[] ins) throws Exception {
        double[] distribution = new double[this.numberOfClusters()];
        for (int i = 0; i < numberOfClusters(); i++) 
		{
            distribution[i] = this.distributions.get(i).density(ins);
        }
        return distribution;
    }
    
    public static double[] DoubleArrayToPrimive(Double[] d) 
	{
        double[] ret = new double[d.length];
        for (int i = 0; i < d.length; i++) 
		{
            ret[i] = d[i];
        }
        return ret;
    }

    public static Double[] doubleArrayToWrapper(double[] d) 
	{
        Double[] ret = new Double[d.length];
        for (int i = 0; i < d.length; i++) 
		{
            ret[i] = d[i];
        }
        return ret;
    }

    public double[] distributionForInstance(Double[] ins) throws Exception {
        return distributionForInstance(FIGMN.DoubleArrayToPrimive(ins));
    }

    public double[] distributionForInstance(Instance ins) throws Exception {
        return distributionForInstance(ins.toDoubleArray());
    }
    
    @Override
    public String toString() {
        return "Delta: " + delta + " Beta: " + beta + " #Clusters: " + distributions.size() + "\n" + distributions.toString();
    }

    @Override
    public void buildClassifier(Instances i) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Enumeration<String> enumerateMeasures() 
	{
        java.util.Vector<String> v = new java.util.Vector<String>();
        v.add("measureNumClusters");
        return v.elements();
    }

    @Override
    public double getMeasure(String measureName) 
	{
        if (measureName == "measureNumClusters") 
			return this.numberOfClusters();
        return Double.NaN;
    }
    

}
