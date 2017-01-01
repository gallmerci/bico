BICO (BIrch using COresets)
=======================

This is a python implementation of the original `BICO <http://ls2-www.cs.uni-dortmund.de/bico/>`
code.

BICO is a fast streaming algorithm for the k-means problem. BICO reduces the
number of input points to a specified amount such that the k-means solution on
the compressed input set is still a high quality solution to the original data set.
BICO combines a tree-like data structure of SIGMOND Test of Time Award winning algorithm BIRCH
with reduction techniques from clustering theory.

----

Applications
=======================
Massive Data Sets
=======================
Running k-means on a very large data set is even nowadays infeasible. BICO can reduce the amount of input points drastically
such that running k-means on the resulting reduced data set is just a matter of seconds. BICO ensures that the solution
is of similar quality as the solution on the original input set.

Compression
=======================
BICO reduces a set of input points to a specified number of points while preserving the k-means solution quality as good
as possible. Thus, BICO is a perfect tool to shrink the size of the data in order to reduce space or bandwidth.