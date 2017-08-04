/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package liac.igmn.gui;

import java.awt.Color;
import java.awt.Component;
import javax.swing.BorderFactory;
import javax.swing.JFormattedTextField;
import javax.swing.JTable;
import javax.swing.SwingConstants;
import javax.swing.table.TableCellRenderer;

public class FormatTable implements TableCellRenderer
{
    @Override
    public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int column) 
    {
        JFormattedTextField campoTexto = new JFormattedTextField();
        campoTexto.setBorder(BorderFactory.createEmptyBorder());
        
        if(value instanceof String)
        {
            campoTexto.setHorizontalAlignment(SwingConstants.CENTER);
            campoTexto.setText((String)value);
        }
        if(value instanceof Integer)
        {
            campoTexto.setText(value + "");
            campoTexto.setHorizontalAlignment(SwingConstants.CENTER);
        }
        if(value instanceof Double)
        {
            Double valor = (Double)value;
            campoTexto.setFormatterFactory(new javax.swing.text.DefaultFormatterFactory(new javax.swing.text.NumberFormatter(new java.text.DecimalFormat("0.0000")))); 
            campoTexto.setHorizontalAlignment(SwingConstants.TRAILING); 
            campoTexto.setValue(valor);
        }       
        
        if(isSelected)
        { 
            campoTexto.setBackground(Color.lightGray); 
        }        
        
        return campoTexto;
    }
    
}