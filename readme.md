Search by triplet
=================

Welcome to the search by triplet algorithm written in CUDA.

Here is some documentation for the algorithm idea implemented here:

* https://cernbox.cern.ch/index.php/s/5suOgHdcZmtfpuc

How to run it
-------------

The project requires a graphics card with CUDA support. The build process doesn't differ from standard cmake projects:

    ```mkdir build
    cmake ..
    make```

In order to run it, some binary input files are included with the project. A run of the program with no arguments will let you know the basic options:

    ```Usage: ./cuForward <folder containing .bin files> <number of files to process>```

Here are some example run options:

    ```# Run all input files once
    ./cuForward ../input 50

    # Run a total of 1000 events, round robin over the existing ones
    ./cuForward ../input 1000```
