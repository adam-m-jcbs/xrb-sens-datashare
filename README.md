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
`xrb-sens-datashare`.  To get an updated version of the data, simple `cd` into
where you cloned and execute `git pull`.  This will fetch and add any new files
to your local directory.

If using `git` isn't convenient for you but you're fine to use common \*nix
shell commands, you can get data from the `raw` url for this project
([https://raw.githubusercontent.com/adam-m-jcbs/xrb-sens-datashare/master/](https://raw.githubusercontent.com/adam-m-jcbs/xrb-sens-datashare/master/)).

For example,
```
wget https://raw.githubusercontent.com/adam-m-jcbs/xrb-sens-datashare/master/README.md
```
will download this `README.md` file to your local working directory.  As
resources are added, I will add URLs and brief descriptions below.

## Key Available Raw Data: Python Dictionary Pickles

One of the features of this data repository is that it lets you
the user interact with and explore our data programatically.
This is made possible by us shipping the data in a form any
computer with Python can extract and analyze.

For those that only want reduced, analyzed data, skip to later
sections.  If you would like to directly load and explore our raw
data, continue.

Now that you have the git repository, `cd` into the `raw_pickles`
directory. From there, execute the
`get_gs1826_10x_grid_data.Nov5.pk.sh` script.  This will use
`wget` to download a copy of the full grid dataset.  These can be
5-10GB and are hosted on public resources, so it may take a while
to download.  Fortunately, you should only need to do this once.

Upon successful download, you will have a file in the directory
called `gs1826_10x_grid_data.Nov5.pk`.  This is basically a
binary version of a Python dictionary full of xrb sensitivity
data.  In later sections, we will see how to load this data and
reduce it using the same scripts the authors of this repo use to
reduce our data from these raw sources.

One snag you will run into is that the objects in this dictionary
have some third-party dependencies.  Not to fret, this repository
will provide all needed dependencies.

## Key Available Data

**`gs1826/ash/gs1826_100x_ash_metrics_tab.dat`**  
Copy with  
```
wget https://raw.githubusercontent.com/adam-m-jcbs/xrb-sens-datashare/master/gs1826/ash/gs1826_100x_ash_metrics_tab.dat
```  
[Ash data details](gs1826/ash/ash_data_overview.md)

## Credit / Authorship

As a living repository, many are involved in the creation, maintenance,
analysis, and data generation.  A more formal and systematic policy for
contributing and giving due credit will be implemented in this repository.  In
the meantime, the following are the original authors, along with some of their
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
