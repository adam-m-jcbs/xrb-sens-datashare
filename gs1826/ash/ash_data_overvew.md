# Ash Data Overview

This document gives an overview of the ash data and how to read it.

## Ash tables

Ash data is exported as a structured ASCII table, giving a balance of human and
machine readability as well as trivial portability.

Currently available tables include
```
gs1826/ash/gs1826_100x_ash_metrics_tab.dat
```

#### Reading the data - an example

To help orient you to the data, we'll read an example.

The head of `gs1826_100x_ash_metrics_tab.dat` gives
```
  A Z ab (UvD|DvU)   |        A_ash         |    change_factor     |      change_min      |      change_max      |      change_mag      |       mag_min        |       mag_max        | 
========================================================================================================================================================================
    69 35 gp DvU     |         107          |  2.089837191185071   |  1.656281724779846   |  2.520236664371889   | 0.0004223805035136898 | 0.000284893568294053 | 0.0004967866187289429 | 
```

The headers are defined as
+ `A Z ab (UvD|DvU)`: The A and Z of the "target nucleus" in experimental
  parlance. `ab` gives the "channel" of the reaction as in X(a,b)Y . `UvD`
  indicates that the sensitivity is measured by comparing the up variation model
  against the corresponding down variation model ("Up vs Down"). `DvU` is the
  opposite.  I felt it important to include this since we're comparing _not_
  with a baseline but with the opposite variation.  For lightcurves, we compare
  against a baseline.  
+ `A_ash`: The mass number `A` of the ash being analyzed in this line.  All
  impacts in subsequent columns are for this specific ash.
+ `change_factor`:  The factor by which `X_A` changed, where `X_A` is the mass
  fraction of ashes with mass number A.  Thus, `change_factor = X_Ad / X_Au` for a
  `UvD` line.  This change is based on taking an average of `X_A` over the
  region where ashes develop.  
+ `change_min`/`change_max`: The minimum and maximum values of this change in
  the ash region.  This gives a measure of how noisy data may be for this ash.
+ `change_mag`: The magnitude by which `X_A` changed.  Thus, 
  `change_mag = X_Ad - X_Au` for a `UvD` line.  Again, `X_A` is averaged over
  the ash region.
+ `mag_min`/`mag_max`: The minimum and maximum values of this ash's magnitude
  change in the ash region.

So the first line we saw above tells us several things.  First, let's determine
the reaction from it.  `A,Z=69,35` tells us the target nucleus is 69Br.  It's a
(g,p) reaction.  And we're comparing the model that varies this reaction rate
down against the model that varies it up (`DvU`).  

In my own system (I really don't think of these things the way nuclear
experimentalists seem to), I call this model `br69gp0.01` to show this model is
based on varying the `br69gp` rate down by a factor of 100.

According to my model data, `br69gp0.01`'s `<X_107>` is 2.0898 times larger than
that of `br69gp100.0`.

According to my model data, `br69gp0.01`'s `<X_107>` is 0.0004 smaller than
that of `br69gp100.0` (`change_mag = X_down - X_up` for `DvU`).

TODO: Need to add plots to make this clearer, and might want to discuss with
collaborators potential changes to make to definitions.  This data is quite hard
to clearly reduce into metrics like this.  But we can't give a table of 2500
plots, so here we are...

#### Reading the data - label conventions

"100x" grids are those with variations up and down by a factor of 100.  Similar
for "10x" grids.

Models/grids using gs1826 as a baseline include "gs" or "gs1826"

Models/grids using 4u1820 as a baseline include "4u" ("Fu" if forced by name
restrictions) or "4u1820"

## Note on Python pickle files

If you're more technically inclined and prefer to access/manipulate the grid
data programmatically the same way I do, ask me about a Python pickle that
contains all of the raw data from entire grids accessible through a nested
dictionary.  Pickle files can be GBs big, so aren't feasible for this repo, but
be aware that these pickle files are the ultimate source for all reduced data in
this repo.
