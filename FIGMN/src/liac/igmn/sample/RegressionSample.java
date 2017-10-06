/**
 * =============================================================================
 * Federal University of Rio Grande do Sul (UFRGS)
 * Connectionist Artificial Intelligence Laboratory (LIAC)
 * Jorge C. Chamby Diaz - jccdiaz@inf.ufrgs.br
 * =============================================================================
 * Copyright (c) 2017 Jorge C. Chamby Diaz, jchambyd at gmail dot com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * =============================================================================
 */
package liac.igmn.sample;

import org.ejml.simple.SimpleMatrix;
import com.panayotis.gnuplot.JavaPlot;
import com.panayotis.gnuplot.plot.DataSetPlot;
import com.panayotis.gnuplot.style.Style;
import liac.igmn.core.IGMN;
import liac.igmn.util.MatrixUtil;

public class RegressionSample {

	public static void main(String[] args)
	{
		// gen some data
		int i = 0;
		SimpleMatrix dataset = new SimpleMatrix(63, 2);
		for (float x = 0; x <= 2 * Math.PI; x += 0.1f) {
			dataset.set(i, 0, x);
			dataset.set(i, 1, Math.sin(x));
			i++;
		}

		// plot sin
		JavaPlot p = new JavaPlot();

		p.setTitle("sin");
		p.getAxis("x").setLabel("X axis");
		p.getAxis("y").setLabel("Y axis");

		DataSetPlot datasetPlot = new DataSetPlot(MatrixUtil.toDouble(dataset));
		datasetPlot.getPlotStyle().setStyle(Style.LINES);
		datasetPlot.setTitle("sin(x)");
		p.addPlot(datasetPlot);

		// model setup
		SimpleMatrix range = new SimpleMatrix(new double[][]{{2, 2}});
		IGMN igmn = new IGMN(range.transpose(), 0.1, 0.3);

		// train model
		igmn.train(dataset.transpose());

		// test model
		SimpleMatrix out = dataset.copy();
		dataset = dataset.extractMatrix(0, SimpleMatrix.END, 0, 1);
		for (i = 0; i < dataset.numRows(); i++) {
			SimpleMatrix y = igmn.recall(dataset.extractVector(true, i));
			out.set(i, 1, y.get(0));
		}

		// plot regression
		datasetPlot = new DataSetPlot(MatrixUtil.toDouble(out));
		datasetPlot.getPlotStyle().setStyle(Style.LINES);
		datasetPlot.setTitle("igmn");
		p.addPlot(datasetPlot);

		p.plot();
	}
}
