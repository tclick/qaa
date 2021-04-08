Quick Start
===========
Quasi-anharmonic analysis (QAA) is an additional technique to analyze molecular
dynamics (MD) trajectories. It can be used with various topology and trajectory
formats including `Amber`_, `CHARMM`_, and `Gromacs`_.

A typical process involves four steps:

#) Alignment of the trajectory with its average structure

#) Determination of number of components by principal component analysis (PCA)

#) Calculation of the unmixed signals using spatially-independent ICA

#) Analysis and visualization of the data using clustering techniques

Alignment
---------
We assume that the user has already stripped the trajectory of solvent and any
extraneous groups that may be irrelevant to the trajectory. Furthermore, the
user will have aligned the trajectory within the periodic boundary conditions so
as to retain a whole molecule.

We align the trajectory to its average structure simply as a way to center the
trajectory. In this example, we also reduce the trajectory to its
C:math:`{\alpha}` atoms only.

.. code-block:: bash

    qaa align -s protein.parm7 -f protein.nc -r average.pdb -o align.nc -m ca \
        -l align_traj.log -v

This command will read the given topology and trajectory files, and save both
the average structure file (:code:`-r`) and the aligned trajectory file
(:code:`-o`). In the logged output, we will see how many iterations were needed
to align the trajectory within a reasonable approximation to the average
structure.

PCA
---
Now that the trajectory has been aligned, we are ready to determine a reasonable
number of components to use with QAA. We accomplish this by performing principal
component analysis (PCA) on our recently-aligned trajectory.

.. code-block:: bash

    qaa pca -s average.pdb -f align.nc -l pca.log -w --bias --image -v

By using the :code:`-w` and :code:`--bias`, we are projecting the data onto a
phase-space using :math:`\sqrt{N}` To use an unbiased population of
:math:`\sqrt{N-1}`, simply omit :code:`--bias`. The log file will include
information on the percentage of the explained variance covered with a selection
of various numbers of components. This will include percentages between
:math:`75-95\%` variance (increments of 5%) as well as percentages when 50 and
100 components are selected.

:code:`--image` writes two image files:

#) the explained variance ratio (similar to a Scree plot)

#) the projected values of the first three components both as three 2D plots and a 3D plot

Additionally, the coordinate projection, singular values, explained variance,
and components are written in both the CSV and NumPy binary formats. The CSV
format allows for easy portability to other programs like Matlab or Excel. The
NumPy binary format offers more precise data by retaining the full
floating-point values.

QAA
---
Once the number of components have been decided, QAA can be initiated.

.. code-block:: bash

    qaa qaa -s protein.parm7 -f protein.nc -l qaa-jade.log --jade -n 50 -v --image

In this case, we reduce the trajectory down to 50 components for analysis. We
also chose to use a 4th-order ICA method (joint diagonalization), but we could
have also selected FastICA using a 3rd order (cubic) equation. FastICA typically
will converge faster than Jade, but Jade will sort its unmixing matrix.

The :code:`--image` option will save the projected values for the first three
components. It is similar to the PCA option in which three 2D and one 3D plot
are displayed.

Similarly to PCA, the projection and unmixing matrix data are saved both in the
CSV and NumPy binary formats for future usage.

Cluster Analysis
----------------
Finally, the user can further analyze the data produced either by PCA or QAA
using cluster analysis techniques. :code:`qaa cluster` currently provides two
clustering methods: Gaussian mixture models and k-means.

.. code-block:: bash

    qaa cluster -s protein.parm7 -f protein.nc -i qaa-signals.npy -o cluster-jade.png -l cluster-jade.log -v --save

Similar to PCA and QAA, the image file will contain both three 2D and one 3D
figures. In this case, it will be of the first three components, but other
dimensions can be selected using the :code:`--axes` option. The images will
visualize the clusters in different colors and display the average for each
cluster as well.

By including the :code:`--save` option, the coordinates of the trajectory
nearest the average structure are also saved in a PDB format. This allows
further analysis of relevant structures within each cluster using a
visualization program like `PyMol`_ or `VMD`_.

.. _Amber: http://www.ambermd.org/
.. _CHARMM: https://www.charmm.org/
.. _Gromacs: http://www.gromacs.org/
.. _PyMol: https://www.pymol.org/
.. _VMD: https://www.ks.uiuc.edu/Research/vmd/
