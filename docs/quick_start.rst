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

    qaa pca -s average.pdb -f align.nc -l pca.log -w --it png -v

By using the :code:`-w` and :code:`--bias`, we are projecting the data onto a
phase-space using :math:`\sqrt{N}` To use an unbiased population of
:math:`\sqrt{N-1}`, simply omit :code:`--bias`. The log file will include
information on the percentage of the explained variance covered with a selection
of various numbers of components. This will include percentages between
:math:`75-95\%` variance (increments of 5%) as well as percentages when 50 and
100 components are selected.

The option :code:`--it png` provides the output style for the explained variance
ratio. This allows the user to visualize the curve of \% explained variance
within each component, which can assist in determining the number of components
to use when running QAA.

Additionally, the coordinate projection, singular values, explained variance,
and components are written in both the CSV and NumPy binary formats. The CSV
format allows for easy portability to other programs like Matlab or Excel. The
NumPy binary format offers more precise data by retaining the full
floating-point values.

QAA
---
Once the number of components have been decided, QAA can be initiated.

.. code-block:: bash

    qaa qaa -s average.pdb -f align.nc -l qaa-jade.log --jade -n 50 -v

In this case, we reduce the trajectory down to 50 components for analysis. We
also chose to use a 4th-order ICA method (joint diagonalization), but we could
have also selected FastICA using a 3rd order (cubic) equation. FastICA typically
will converge faster than Jade, but Jade will sort its unmixing matrix.

Similarly to PCA, the projection and unmixing matrix data are saved both in the
CSV and NumPy binary formats for future usage.

Cluster Analysis
----------------
Finally, the user can further analyze the data produced either by PCA or QAA
using cluster analysis techniques. :code:`qaa cluster` currently provides two
clustering methods: Gaussian mixture models and k-means.

.. code-block:: bash

    qaa cluster -s average.pdb -f align.nc -i qaa-signals.csv --ica -l qaa-cluster.log --iter 1000 --dp 5 -n 4 --save

Similar to PCA and QAA, the image file will contain both three 2D and one 3D
figures. In this case, it will be of the first three components, but other
dimensions can be selected using the :code:`--axes` option. The images will
visualize the clusters in different colors and display the average for each
cluster as well.

By including the :code:`--save` option, the coordinates of the trajectory
nearest the average structure are also saved in a PDB format. This allows
further analysis of relevant structures within each cluster using a
visualization program like `PyMol`_ or `VMD`_.

Visualization
-------------
Through the process, we can visualize the projection data on both a 2D and 3D
plane. To visualize the combined 2D and 3D projections, one simply runs

.. code-block:: bash

    qaa plot -i ica-cluster.csv -o ica-cluster.png -l ica-plot-cluster.log -c ica-centroids.csv --ica -v

This will create comparison plots of the three components and a subsequent 3D
plot. The user can additionally adjust the azimuth (z-axis rotation) and the
elevation of the 3D plot.

If the user has clustered data, you simply add :code:`--cluster` to the above
command

.. code-block:: bash

    qaa plot -i ica-cluster.csv -o ica-cluster.png -l ica-plot-cluster.log -c ica-centroids.csv --ica --cluster -v

and the plot will colorize the clusters for enhanced visualization.

For additional visualization examples, go to `Github`_ and look at the
notebooks subdirectory. Using `Holoviews`_ in a Jupyter notebook, one can
interactively visualize the data. The included notebooks offer tutorials, and
the visualization code cells can be copied and modified to work with your data.

.. _Amber: http://www.ambermd.org/
.. _CHARMM: https://www.charmm.org/
.. _Github: https://www.github.com/tclick/qaa/
.. _Gromacs: http://www.gromacs.org/
.. _Holoviews: https://www.holoviews.org/
.. _PyMol: https://www.pymol.org/
.. _VMD: https://www.ks.uiuc.edu/Research/vmd/
