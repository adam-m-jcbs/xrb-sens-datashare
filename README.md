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

## Key Available Data

**`gs1826/ash/gs1826_100x_ash_metrics_tab.dat`**  
Copy just this file with  
```
wget https://raw.githubusercontent.com/adam-m-jcbs/xrb-sens-datashare/master/gs1826/ash/gs1826_100x_ash_metrics_tab.dat
```  
This file contains the results of analyzing a grid of ~1200 reaction rate
variations by a factor of 100 up and down.  The data is in terms of ash metrics
we've developed.  TODO: add more explanation.

