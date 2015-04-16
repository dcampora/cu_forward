Memory required on the GPU at this point per event:


Custom datatypes + input datafile in MiB:

(2 * 10000 * 116 + 3 * 10000 * 4 + 61 * 1024) / (1024.0 * 1024.0) = 2.4 MiB per event

---

TODO:

After the ameising conversation with D. Rohr :)

- Do an array of neighbours to search from at the beginning, per hit.
  This way, we'll have coalescence (!) when searching in the window of the hit :)
  We don't even need a dynamic array for the hits - Since only the first two hits
  are checked for the neighbours.

That should be all :)
