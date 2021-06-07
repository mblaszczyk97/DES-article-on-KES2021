import pandas as pd
import numpy as np

class irregular_set():
    """
    class representing irregular dataset

    Attributes
        ----------
        location : str
            name of the file with extension ex. "irregularData1.csv"

        db : dataframe
            dataframe of irregular set

        X : dataframe
            database without class attribute

        y : dataframe
            database with only class attribute
    """

    def __init__(self, location):
        """
        Parameters
        ----------
        location : str
            name of the file with extension ex. "irregularData1.csv"
        """
        self.location = location
        self.db = pd.read_csv('../input/bases/{}'.format(self.location))
        self.y = self.db.Class
        del self.db["Class"] 
        self.X = self.db.values.astype(np.float)

    def get_attributes_number(self):
        """returns numer of attributes in set without class attribute"""
        return  len(self.db.columns)-1

    def get_records_number(self):
        """returns how many records is in set"""
        return  len(self.db)-1

    def get_imbalanced_ratio(self):
        """returns IMBALANCED RATIO which is numer of minority class compared to number majority class"""
        y = self.y
        negativeClassName=' negative'
        positiveClassName=' positive'
        imbRatio=(sum(y.values.ravel()==negativeClassName))/(sum(y.values.ravel()==positiveClassName))
        if y.values[1]==negativeClassName.strip() or y.values[1]==positiveClassName.strip():
            imbRatio=(sum(y.values.ravel()==negativeClassName.strip()))/(sum(y.values.ravel()==positiveClassName.strip()))  
        return imbRatio
