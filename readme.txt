Memory required on the GPU at this point per event:


Custom datatypes + input datafile in MiB:

(2 * 10000 * 116 + 3 * 10000 * 4 + 61 * 1024) / (1024.0 * 1024.0) = 2.4 MiB per event


---

Using 32 threads in flight, 1 block

Max: 2048 CUDA threads in this machine

---

Let's change to 64 threads per block and issue 128 blocks (128 events in parallel)

---

z from hit RE values:

PrChecker.Velo       INFO **** Velo                            66650 tracks
including           612 ghosts [ 0.9 %], Event average   0.7 % ****
PrChecker.Velo       INFO                             velo :   56090 from
63129 [ 88.8 %]   3478 clones [ 5.8 %], purity: 99.86 %, hitEff: 91.07 %
PrChecker.Velo       INFO                             long :   19617 from
19771 [ 99.2 %]   1728 clones [ 8.1 %], purity: 99.86 %, hitEff: 90.14 %
PrChecker.Velo       INFO                        long>5GeV :   12946 from
12996 [ 99.6 %]   1168 clones [ 8.3 %], purity: 99.86 %, hitEff: 90.36 %
PrChecker.Velo       INFO                     long_strange :     831 from
845 [ 98.3 %]     45 clones [ 5.1 %], purity: 99.52 %, hitEff: 93.19 %
PrChecker.Velo       INFO                long_strange>5GeV :     400 from
408 [ 98.0 %]     20 clones [ 4.8 %], purity: 99.44 %, hitEff: 94.32 %
PrChecker.Velo       INFO                       long_fromB :     819 from
827 [ 99.0 %]     63 clones [ 7.1 %], purity: 99.75 %, hitEff: 90.98 %
PrChecker.Velo       INFO                  long_fromB>5GeV :     677 from
680 [ 99.6 %]     55 clones [ 7.5 %], purity: 99.71 %, hitEff: 90.71 %


