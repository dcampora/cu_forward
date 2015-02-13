Memory required on the GPU at this point per event:


Custom datatypes + input datafile in MiB:

(2 * 10000 * 116 + 3 * 10000 * 4 + 61 * 1024) / (1024.0 * 1024.0) = 2.4 MiB per event


---

Using 32 threads in flight, 1 block

Max: 2048 CUDA threads in this machine

---

Let's change to 64 threads per block and issue 128 blocks (128 events in parallel)

