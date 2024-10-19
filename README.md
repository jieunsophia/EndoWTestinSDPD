# EndoWTestinSDPD
Replication package for 'Testing the Endogeneity of a Spatial Weight Matrix in the Weak-Tied Spatial Dynamic Panel Data Model'


# About
- Paper title: Testing the Endogeneity of a Spatial Weight Matrix in the Weak-Tied Spatial Dynamic Panel Data Model
- Author: Jieun Lee, Postdoctoral Fellow at the Hubert Department of Global Health, Emory University Rollins School of Public Health
- This work was conducted during my PhD in Economics at the University of Illinois Urbana-Champaign and continued during my postdoctoral training at the Emory University Rollins School of Public Health.


# Replications
- The results in Table 2 can be retrieved from the files in the '[Supplement 2] Simulation_Initial_conditions_problem' folder.

- The results in Tables 3 to 9 & Figures 1 to 6 & Figures A.2 to A.3 & OA.1 can be retrieved from the files in the '[Main 1] Simulation_RStest' and '[Main 2] Simulation_C(a)test' folders.

- The results in Tables 10 and 11 can be retrieved from the files in the '[Main 3] Empirical_application' folder.

- The results in Tables A.1 to A.2 & Figure A.1 can be retrieved from the files in the '[Supplement 1] Simulation_Effects_weak_sdpd' folder.


# Some Notes
info = struct('n',n,'t',t,'rmin',0,'rmax',1,'lflag',0,'tl',1,'stl',1,'tlz',1,'tly',0); 

tl means if we include time lag of Y in the main equation. <br>
stl means if we include spatial time lag of Y in the main equation. <br>
tlz means if we include time lag of Z in the auxiliary equation. <br>
tly means if we include time lag of Y in the auxiliary equation. <br>


# Questions
If you have any questions, please feel free to reach me out (jieun.lee@emory.edu; jieunlee.sophia@gmail.com). I'm happy to answer your questions.


Thank you, <br>
Jieun Lee
