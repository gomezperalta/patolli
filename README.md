# patolli

patolli was a typical game in ancient Mexico. 
Due to its predictive attributes on the fate of the people, patolli had a ritual character.
With patolli.py you can create your own patollis, which will help you to infer new materials with a crystal structure.

The way the patollis are created is with Artificial Neuronal Networks (ANNs). These ANNs are trained as a binary classification model: given a crystal structure, the ANNs learn to classify the compounds as True or False samples.

patolli.py needs the next Python modules:
<ul>
  <li>keras</li>
  <li>scikit-learn</li>
  <li>numpy</li>
  <li>pandas</li>
  <li>matplotlib</li>  
  <li>itertools</li>
  <li>copy</li>
  <li>time</li>
  <li>os</li>
</ul>

In your conda environment, pydot or graphviz must be installed. patolli.py was developed in Python 3.6.

When you run patolli.py, you will have to provide the name of the next txt-files:
<ul>
  <li>structure_dictionary: The definition of the crystal structure in terms of occupied Wyckoff sites.</li>
  <li>model_control_file: The hyperparameters of the ANNs you will train.</li>
</ul>

You should not remove the files within the directory 'support'. Otherwise, patolli.py crashes.

This program is part of my Ph.D. thesis. I thank to CONACyT (Scholarship No. 336003), the AI - Lab of Institute of Physics, UNAM, and to Dr. Xim Bokhimi for his guidance.

If you benefit of this code, I appreciate that you cite the next article:

Juan I. Gómez-Peralta, X. Bokhimi. Journal of Solid State Chemistry, Vol. 285 (2020) 121253.
