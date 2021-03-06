# xrb-sens-datashare
A repository for sharing intermediate and final data products in machine and/or
human-readable formats coming out of the JINA-CEE xrb reaction rate sensitivity
project.

DISCLAIMER: The primary author and maintainer of this repository
is Adam Jacobs.  Any mistakes, confusing data, wrong data, or
similar flaws in this repo are my responsibility and mine alone.
Please address questions and issues to me, not to those that may
maintain any of the tools/codes I have used.  Any amazing things
in the repo feel free to attribute to my fantastic collaborators.

## Getting Data

If you're familiar with `git`, then you can easily get the data with the usual git workflow:

```
git clone git@github.com:adam-m-jcbs/xrb-sens-datashare.git
```

This repo is open, so no credentials or account should be needed for read (clone) access.

The above command will copy the files into a `git`-managed directory called
`xrb-sens-datashare`.  To get an updated version of the data, simply `cd` into
where you cloned and execute `git pull`.  This will fetch and add any new files
to your local directory.  Note that because of their size, raw
datasets must be manually fetched using the scripts provided.

If using `git` isn't convenient for you but you're fine to use common \*nix
shell commands, you can get some data from the `raw` url for this project
([https://raw.githubusercontent.com/adam-m-jcbs/xrb-sens-datashare/master/](https://raw.githubusercontent.com/adam-m-jcbs/xrb-sens-datashare/master/)).

For example,
```
wget https://raw.githubusercontent.com/adam-m-jcbs/xrb-sens-datashare/master/README.md
```
will download this `README.md` file to your local working directory.  As
resources are added, I will add URLs and brief descriptions below.

## Key Available Raw Data Products: Python Dictionary Pickles

One of the features of this data repository is that it lets the
user interact with and explore our _raw_ data programatically in
Python, in addition to providing our own reduced analysis of the
raw data.  This is made possible by us shipping the data in a
form any computer with Python can extract and analyze.

For those that only want reduced, analyzed data, skip to later
sections.  If you would like to directly load and explore our raw
data (with no promises of completeness or correctness), continue.

Now that you have the git repository, `cd` into the `raw_pickles`
directory. From there, execute the
`get_gs1826_10x_grid_data.Nov5.pk.sh` script.  This will use
`wget` to download a copy of the full `gs1826_10x` grid dataset.  These can be
5-10GB and are hosted on public resources, so it may take a while to download.
Fortunately, you should only need to do this once.

Upon successful download, you will have a file in the directory
called `gs1826_10x_grid_data.Nov5.pk`.  This is basically a
binary version of a Python dictionary full of xrb sensitivity
data.  

As of now, this data structure requires some packages before it can be loaded.
For convenience, we directly provide those packges here.  The `load_pickle.py`
script gives an example of loading a grid of data out of the pickle and into a
nested dictionary (which tries to be self-documenting).

This script can serve as a starting template for engaging with our raw data.

Some python dependencies will be needed, including
- matplotlib
- numpy-quaternion
- scipy


## Key Available Reduced Data Products

This section provides a convenient overview of available reduced data, links to
guides for working with the different kinds of reduced data (and how to
generate your own), and shell commands for downloading them

[Ash data overview](gs1826/ash/ash_data_overview.md)

**`gs1826/ash/gs1826_100x_ash_metrics_tab.dat`**  
A collection of reduced model results measuring the impact of reaction rate
variations on the ash profile generated by bursts.  
About 2500 Kepler models.  
The models are variations over a GS 1826 baseline, with rp-process-path
reactions varied up and down by a factor of 100.
```
wget https://raw.githubusercontent.com/adam-m-jcbs/xrb-sens-datashare/master/gs1826/ash/gs1826_100x_ash_metrics_tab.dat
```  

## Credit / Authorship

As a living repository, many are involved in the creation, maintenance,
analysis, and management.  A more formal, systematic, and fair policy for
contributing and giving due credit will be implemented in this
repository.  In the meantime, the following are the original
authors, along with some of their
primary contributions:

Adam Jacobs (Maintainer, Original Author):  
JINA-CEE, MSU  
  + Developed the vast majority of the infrastructure for carrying out simulations and managing their data
  + Deployed and managed the Kepler simulation suites.  Currently at about 10,000 archived models and counting.
  + Coordinated and organized the collaboration

Zac Johnston (Original Author):  
MSU  
  + Contributed essential components to the simulation/data management framework
  + Provided expert consultation on Kepler, carrying out massive 1D parameter studies, statistical analysis

Ed Brown & Hendrik Schatz (Original Authors):  
JINA-CEE, MSU, NSCL, FRIB
  + Secured funding and managed logistics / people
  + Provided expert consultation on Kepler, nuclear theory, nuclear experiment, nuclear data, nuclear astrophysics
  + Organized many collaborative meetings across the globe at leading institutions in nuclear astrophysics

Matt Amthor (Original Author):  
Bucknell
  + Provided expert consultation on the preceding study, which he was a co-author of
  + Provided expert consultation on nuclear experiment, Kepler, and data anlaysis
  + Carried out analysis of reaction network flows

Alexander Heger (Original Author):  
Monash
  + Provided access to and support for using the Kepler stellar evolution code
  + Actively maintains and enhances the Kepler stellar evolution code, including to facilitate this simulation suite's efforts
  + Hosted collaborative meetings at Monash
  + Provided expert consultation on Kepler, nuclear theory, nuclear data, nuclear astrophysics

## LICENSE / Copyright

License details are pending.  As of now, all of this is  
copyright JINA-CEE / MSU 2020  
with all rights reserved.  An open, citable release is in the works.  This
should be considered as a read-only reference for the time-being.
