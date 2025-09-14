import unittest
import numpy as np
from L00 import esCuadrada  

class TestEsCuadrada(unittest.TestCase):
    
    # 0x0
    def test_vacia(self):   
        matrix = np.array([])
        self.assertTrue(esCuadrada(matrix))  
    
    # 2x2
    def test_cuadrada(self):  
        matrix = np.array([[1,2],[3,4]])
        self.assertTrue(esCuadrada(matrix))
    
    # 2x3
    def test_mas_col_que_fil(self):  
        matrix = np.array([[1,2,3],[4,5,6]])
        self.assertFalse(esCuadrada(matrix))

    # 3x2
    def test_mas_fil_que_col(self):
        matrix = np.array([[1,2],[3,4],[5,6]])
        self.assertFalse(esCuadrada(matrix))

    # Un solo elemento
    def test_un_elemento(self):  
        matrix = np.array([1])
        self.assertFalse(esCuadrada(matrix))