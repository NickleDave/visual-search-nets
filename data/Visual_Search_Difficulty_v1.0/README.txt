————————————————> 1. License agreement

 Copyright (C) 2016  Radu Tudor Ionescu, Bogdan Alexe, Marius Leordeanu,
 Marius Popescu, Dimitrios Papadopoulos, Vittorio Ferrari
 
 This package contains free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the Free Software
 Foundation, either version 3 of the License, or any later version.
 
 This program is distributed in the hope that it will be useful, but WITHOUT ANY
 WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 PARTICULAR PURPOSE. See the GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License along with this
 program (see COPYING.txt package file). If not, see <http://www.gnu.org/licenses/>.


————————————————> 2. Citation
 
 Please cite the corresponding work (see citation.bib package file to obtain the
 BibTex) if you use this data set in any scientific work:

 [1] Radu Tudor Ionescu, Bogdan Alexe, Marius Leordeanu, Marius Popescu, 
     Dimitrios Papadopoulos, Vittorio Ferrari. How hard can it be? Estimating the 
     difficulty of visual search in an image. Proceedings of CVPR, pp. 2157-2166,
     2016.


————————————————> 3. Visual Search Difficulty Data Set and Code Website:

 This data set is available at: http://image-difficulty.herokuapp.com/


————————————————> 4. Format details
 
 The scores associated to images from the Pascal VOC 2012 train and validation set
 are provided in CSV and Matlab formats. 

 In VSD_dataset.csv is structured at follows:
 - the first column contains the names of the image files in Pascal VOC 2012;
 - the second column contains the associated difficulty scores.

 The VSD_dataset.mat contains two Matlab variables:
 - names => contains the names of the image files in Pascal VOC 2012;
 - scores => contains the associated difficulty scores (e.g.: names(i) has the
 associated score given at scores(i)).


————————————————> 5. Feedback and suggestions
 
 Send an e-mail to: raducu[dot]ionescu{at}gmail[dot].com
