import math, json, unittest
import agrosearch_app as agro
class TestAgro(unittest.TestCase):
 def test_normalizacao_e_controles(self):
  self.assertEqual(agro.preprocessar('A IRRIGAÇÃO!',False,False),['a','irrigacao'])
  self.assertEqual(agro.preprocessar('A IRRIGAÇÃO!',True,False),['irrigacao'])
  self.assertNotEqual(agro.preprocessar('lagartas',True,True),['lagartas'])
 def test_indice_e_pesos_conhecidos(self):
  ts,indice,idf,vet=agro.construir_indice({'a':'soja soja agua','b':'agua milho'},False,False)
  self.assertEqual(indice['agua'],['a','b']); self.assertEqual(indice['soja'],['a'])
  self.assertAlmostEqual(idf['agua'],1); self.assertAlmostEqual(vet['a']['soja'],2/3*(math.log(3/2)+1))
 def test_cosseno_e_vazios(self):
  self.assertEqual(agro.cosseno({},{}),0)
  self.assertAlmostEqual(agro.cosseno({'a':2},{'a':9}),1)
  for q in ['', 'e a de','inexistentezz']:
   linhas,_,_,c=agro.buscar(q)
   self.assertFalse(c); self.assertTrue(all(r['TF-IDF acumulado']==0 and r['Cosseno']==0 for r in linhas))
 def test_ranking_exato(self):
  r,*_=agro.buscar('Trichogramma')
  self.assertEqual(r[0]['Documento'],'Doc 2')
  for sw in [True,False]:
   for stem in [True,False]:
    _,ind,_,_=agro.construir_indice(agro.DOCUMENTOS,sw,stem)
    self.assertTrue(ind)
if __name__=='__main__': unittest.main(verbosity=2)
