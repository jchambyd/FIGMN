package core;

import no.uib.cipr.matrix.DenseMatrix;

public class Matrix extends DenseMatrix {

    public double det;
    
    public Matrix(int rows, int columns) {
        super(rows, columns);
    }
    
    public Matrix(int rows, int columns, double det) {
        super(rows, columns);
        this.det = det;
    }
    
    public Matrix(DenseMatrix m) {
        super(m);
    }
    
    public Matrix(DenseMatrix m, double det) {
        this(m);
        this.det = det;
    }
}
