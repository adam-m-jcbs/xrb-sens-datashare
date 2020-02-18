# xrb-sens-datashare
A repository for sharing intermediate and final data products in machine and/or
human-readable formats coming out of the JINA-CEE xrb reaction rate sensitivity
project.

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
