/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package liac.igmn.gui;

import javax.swing.JOptionPane;
import liac.igmn.core.IGMN;
import org.ejml.simple.SimpleMatrix;

/**
 *
 * @author liac01
 */
public class Interface extends javax.swing.JFrame {

	private IGMN poIGMN;
	private int pnNumInstances;
	private int pnCurrent;
	private SimpleMatrix paDataSet;
	
	
	/**
	 * Creates new form Interface
	 */
	public Interface() {
		initComponents();
		this.mxStateControls(1);
	}

	/**
	 * This method is called from within the constructor to initialize the form. WARNING: Do NOT modify this code. The content of this method is always regenerated by the Form Editor.
	 */
	@SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPanel1 = new javax.swing.JPanel();
        txnNumIns = new javax.swing.JTextField();
        cmbLearn = new javax.swing.JButton();
        txnTau = new javax.swing.JTextField();
        txnDelta = new javax.swing.JTextField();
        jLabel1 = new javax.swing.JLabel();
        jLabel2 = new javax.swing.JLabel();
        jLabel3 = new javax.swing.JLabel();
        cmbInicialize = new javax.swing.JButton();
        jLabel4 = new javax.swing.JLabel();
        txnTotIns = new javax.swing.JTextField();
        cmbExit = new javax.swing.JButton();
        cmbVisualize = new javax.swing.JButton();
        cmbReset = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("IGMN");

        jPanel1.setBorder(javax.swing.BorderFactory.createEtchedBorder());

        txnNumIns.setText("1");

        cmbLearn.setText("Learn");
        cmbLearn.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mousePressed(java.awt.event.MouseEvent evt) {
                cmbLearnMousePressed(evt);
            }
        });

        txnTau.setText("0.1");

        txnDelta.setText("0.3");

        jLabel1.setText("Tau:");

        jLabel2.setText("Delta:");

        jLabel3.setText("# Instances");

        cmbInicialize.setText("Inicialize");
        cmbInicialize.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mousePressed(java.awt.event.MouseEvent evt) {
                cmbInicializeMousePressed(evt);
            }
        });

        jLabel4.setText("Total Instances:");

        txnTotIns.setText("0");
        txnTotIns.setEnabled(false);

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addGap(6, 6, 6)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jLabel1)
                    .addComponent(jLabel2)
                    .addComponent(jLabel3))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(jPanel1Layout.createSequentialGroup()
                        .addComponent(jLabel4)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(txnTotIns))
                    .addGroup(jPanel1Layout.createSequentialGroup()
                        .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addComponent(txnDelta, javax.swing.GroupLayout.DEFAULT_SIZE, 64, Short.MAX_VALUE)
                            .addComponent(txnTau)
                            .addComponent(txnNumIns))
                        .addGap(18, 18, 18)
                        .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(cmbInicialize, javax.swing.GroupLayout.DEFAULT_SIZE, 83, Short.MAX_VALUE)
                            .addComponent(cmbLearn, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))))
                .addContainerGap())
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel1Layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(txnTau, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel1))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(txnDelta, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel2)
                    .addComponent(cmbInicialize))
                .addGap(18, 18, 18)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(txnNumIns, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel3)
                    .addComponent(cmbLearn))
                .addGap(18, 18, 18)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel4)
                    .addComponent(txnTotIns, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        cmbExit.setText("Exit");
        cmbExit.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mousePressed(java.awt.event.MouseEvent evt) {
                cmbExitMousePressed(evt);
            }
        });

        cmbVisualize.setText("Visualize");
        cmbVisualize.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mousePressed(java.awt.event.MouseEvent evt) {
                cmbVisualizeMousePressed(evt);
            }
        });

        cmbReset.setText("Reset");
        cmbReset.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mousePressed(java.awt.event.MouseEvent evt) {
                cmbResetMousePressed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                        .addGap(0, 0, Short.MAX_VALUE)
                        .addComponent(cmbReset)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(cmbVisualize)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(cmbExit, javax.swing.GroupLayout.PREFERRED_SIZE, 75, javax.swing.GroupLayout.PREFERRED_SIZE)))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addGap(18, 18, 18)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(cmbExit)
                    .addComponent(cmbVisualize)
                    .addComponent(cmbReset))
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void cmbExitMousePressed(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_cmbExitMousePressed
        // TODO add your handling code here:
        this.dispose();
    }//GEN-LAST:event_cmbExitMousePressed

    private void cmbLearnMousePressed(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_cmbLearnMousePressed
        // TODO add your handling code here:
		this.mxLearn();
    }//GEN-LAST:event_cmbLearnMousePressed

    private void cmbInicializeMousePressed(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_cmbInicializeMousePressed
        // TODO add your handling code here:
		this.mxInicialize();
    }//GEN-LAST:event_cmbInicializeMousePressed

    private void cmbVisualizeMousePressed(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_cmbVisualizeMousePressed
        // TODO add your handling code here:
		this.mxVisualize();
    }//GEN-LAST:event_cmbVisualizeMousePressed

    private void cmbResetMousePressed(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_cmbResetMousePressed
        // TODO add your handling code here:
		this.mxReset();
    }//GEN-LAST:event_cmbResetMousePressed

	private void mxStateControls(int tnState)
	{
		this.txnNumIns.setEnabled(false);
		this.txnDelta.setEnabled(false);
		this.txnTau.setEnabled(false);
		this.cmbInicialize.setEnabled(false);
		this.cmbLearn.setEnabled(false);
		this.cmbVisualize.setEnabled(false);
		this.cmbReset.setEnabled(false);
		
		switch(tnState)
		{
			case 1://Inicialize
				this.txnDelta.setEnabled(true);
				this.txnTau.setEnabled(true);
				this.cmbInicialize.setEnabled(true);
				this.cmbInicialize.requestFocus();
				break;
			case 2://Test
				this.txnNumIns.setEnabled(true);
				this.cmbLearn.setEnabled(true);
				this.cmbLearn.requestFocus();
				this.cmbReset.setEnabled(true);
				break;
			case 3://Visualize
				this.txnNumIns.setEnabled(true);
				this.cmbLearn.setEnabled(true);
				this.cmbVisualize.setEnabled(true);
				this.cmbVisualize.requestFocus();
				this.cmbReset.setEnabled(true);
				break;
			default:
				break;
		}
	}
	
	private void mxLearn()
	{	
		int lnNumTest = Integer.parseInt(this.txnNumIns.getText());
	
		if(this.pnNumInstances < lnNumTest + this.pnCurrent)
		{
			JOptionPane.showMessageDialog(null, "Instances for testing max.: " + (this.pnNumInstances - this.pnCurrent));
            this.txnNumIns.requestFocus();
            return;
		}
		
		SimpleMatrix laDataTest = this.paDataSet.extractMatrix( 0, this.paDataSet.numRows(), this.pnCurrent, this.pnCurrent + lnNumTest);
		
		this.poIGMN.train(laDataTest);
		
		this.pnCurrent += lnNumTest;
		
		this.txnTotIns.setText("" + this.pnCurrent);
		
		this.mxStateControls(3);
	}	
	
	private void mxInicialize()
	{
		int i = 0;
		double lnTau = Double.parseDouble(this.txnTau.getText());
        double lnDelta = Double.parseDouble(this.txnDelta.getText());
		
		SimpleMatrix range = new SimpleMatrix(new double[][]{{2, 2}});
		this.poIGMN = new IGMN(range.transpose(), lnTau, lnDelta);
		
		this.paDataSet = new SimpleMatrix(2, 63);
		for(float x = 0; x <= 2 * Math.PI; x += 0.1f)
		{
			this.paDataSet.set(0, i, x);
			this.paDataSet.set(1, i, Math.sin(x));
			i++;
		}
		
		this.pnNumInstances = 63;
		this.pnCurrent = 0;		
		this.mxStateControls(2);
	}
	
	private void mxVisualize()
	{
		SimpleMatrix loData = this.paDataSet.extractMatrix( 0, this.paDataSet.numRows(), 0, this.pnCurrent);
		
		Visualize loVisualize = new Visualize(this, true, this.poIGMN, this.poIGMN, loData, 2);
        //loVisualize.setVisible(true);
		loVisualize.mxVisualize(this.poIGMN, 0, 1, 1);
		
		this.mxStateControls(2);
	}
	
	private void mxReset()
	{
		this.mxStateControls(1);
		this.txnNumIns.setText("1");
		this.txnTotIns.setText("0");
		this.pnNumInstances = 0;
		this.pnCurrent = 0;
	}
	
	/**
	 * @param args the command line arguments
	 */
	public static void main(String args[]) {
		/* Set the Nimbus look and feel */
		//<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
		/* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
		 */
		try {
			for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
				if ("Nimbus".equals(info.getName())) {
					javax.swing.UIManager.setLookAndFeel(info.getClassName());
					break;
				}
			}
		} catch (ClassNotFoundException ex) {
			java.util.logging.Logger.getLogger(Interface.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
		} catch (InstantiationException ex) {
			java.util.logging.Logger.getLogger(Interface.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
		} catch (IllegalAccessException ex) {
			java.util.logging.Logger.getLogger(Interface.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
		} catch (javax.swing.UnsupportedLookAndFeelException ex) {
			java.util.logging.Logger.getLogger(Interface.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
		}
		//</editor-fold>

		/* Create and display the form */
		java.awt.EventQueue.invokeLater(new Runnable() {
			public void run() {
				new Interface().setVisible(true);
			}
		});
	}

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton cmbExit;
    private javax.swing.JButton cmbInicialize;
    private javax.swing.JButton cmbLearn;
    private javax.swing.JButton cmbReset;
    private javax.swing.JButton cmbVisualize;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JLabel jLabel4;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JTextField txnDelta;
    private javax.swing.JTextField txnNumIns;
    private javax.swing.JTextField txnTau;
    private javax.swing.JTextField txnTotIns;
    // End of variables declaration//GEN-END:variables
}
