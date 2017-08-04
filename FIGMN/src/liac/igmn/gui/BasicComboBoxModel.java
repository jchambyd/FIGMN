/*
 * Autor: Jorge Cristhian Chamby Diaz
 * Version: 1.0
 * Comment: Model Class for ComboBox
 */
package liac.igmn.gui;

import java.util.Vector;
import javax.swing.ComboBoxModel;
import javax.swing.event.ListDataEvent;
import javax.swing.event.ListDataListener;
/**
 *
 * @author jorge
 */
public class BasicComboBoxModel implements ComboBoxModel 
{
    private Vector<BasicComboBoxModelObject> data = new Vector<BasicComboBoxModelObject>();
    private Vector<ListDataListener> list = new Vector<ListDataListener>();
    private BasicComboBoxModelObject selectedItem;

    public BasicComboBoxModel(Integer taCodigos [], String taItems[])
    {
        for(int i = 0; i < taCodigos.length; i++)
        {
            data.add(new BasicComboBoxModelObject(taCodigos[i], taItems[i]));
        }
    }
    public BasicComboBoxModelObject searchSelectedItem(Integer i) 
    {
        for (BasicComboBoxModelObject o : data) {
            if (i.equals(o.getCodigo())) {
                return o;
            }
        }
        return null;
    }
    public BasicComboBoxModelObject searchSelectedItem(String s) 
    {
        for (BasicComboBoxModelObject o : data) {
            if (s.equals(o.getDescri())) {
                return o;
            }
        }
        return null;
    }
    public void setSelectedItem(Object anItem) 
    {
        selectedItem = anItem instanceof BasicComboBoxModelObject ? (BasicComboBoxModelObject) anItem : null;
        for (ListDataListener l : list) {
            l.contentsChanged(new ListDataEvent(this, javax.swing.event.ListDataEvent.CONTENTS_CHANGED, 0, 0));
        }
    }
    public Object getSelectedItem() 
    {
        return selectedItem;
    }
    public int getSize() 
    {
        return data.size();
    }

    public Object getElementAt(int index) 
    {
        return data.get(index);
    }

    public void addListDataListener(ListDataListener l) 
    {
        list.add(l);
    }

    public void removeListDataListener(ListDataListener l) 
    {
        list.remove(l);
    }

    public Integer getSelectedCodigo() 
    {
        return selectedItem == null ? null : selectedItem.getCodigo();
    }

    public String getSelectedDescri() 
    {
        return selectedItem == null ? null : selectedItem.getDescri();
    }
}

class BasicComboBoxModelObject {

    private Integer codigo;
    private String descri;

    public BasicComboBoxModelObject(Integer codigo, String descri) {
        this.codigo = codigo;
        this.descri = descri;
    }

    @Override
    public String toString() {
        return this.getDescri();
    }

    public Integer getCodigo() {
        return codigo;
    }

    public void setCodigo(Integer codigo) {
        this.codigo = codigo;
    }

    public String getDescri() {
        return descri;
    }

    public void setDescri(String descri) {
        this.descri = descri;
    }
}
