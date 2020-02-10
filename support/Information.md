These files are complementary to patolli.py and must not be deleted.

In this directory you should find the next files:

<ul>
  <li>datosrahm.csv: This file contains infomation about the Pauling electronegativity and the atomic radius published by Rahm-Ashcroft-Hoffmann (2015) for each element.</li>
  <li>red_cod-db.pkl: This is a Pandas DataFrame created with crystal compounds having up to 10 Wyckoff sites. The crystal compounds were taken from Crystallography Open Database. The most important information is the one concerning to occupation of the Wyckoff sites. This information is in the column WyckOcc. </li>
  <li>WyckoffSG_dict.npy: This is a Python dictionary saved as a Numpy object. In fact, this is a dictionary of dictionaries. The main dictionary has three keys:
  <ul>  
    <li>wyckmul: A dictionary that relates the Wyckoff symbol with its multiplicities for each space group.</li>
    <li>wycksym: A dictionary that relates the Wyckoff symbol with its point-group symmetry for each space group.</li>
    <li>general: A dictionary that relates the Wyckoff sites, multiplicity with symbol, with the point-symmetry group. This is done for each space group.</li>
  </ul>
  </li>
  <li>fij_2.0_25_diccio.npy: A Python dictionary saved as Numpy object. It contains the value of the local function for each with Wyckoff site. The local function has a gaussian profile. The dictionary relates a cif with its local function value.</li>
</ul>
