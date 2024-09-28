In current_save.c you can find the parallelized version of llama 2's forward function(and its sub-function) for the snitch processor.

With the right compatibility measures applied in the main, these functions could be substituted in the original run.c generate funtion.

To test
0. (if not updated, on the virtualized machine) add cosf.c, sinf.c, __rem_pio2f.c, __sindf.c and _cosdf.c (and possibly others but it should be it) from https://git.musl-libc.org/cgit/musl/tree/src/math/ to /sw/math/src/math(from root folder).
1. (on local machine) run create.exe to create a Transformer in data.h with random values
2. (on local machine) run forward.exe to create results.h (the correct result given the randomized Transformer)
3. on the virtualized machine, follow the tutorial and create a (target/snitch_cluster/)sw/apps/forward/src folder.
4. copy snitch_forward.c in the created src folder. rename it as forward.c
5. create a sw/apps/forward/data folder (in the target/snitch_cluster/ directory, as before)
6. copy structs.h, data.h and results.h in the created data folder
7. Modify the Makefiles to add the forward function to the compiled ones.
8. run 'make DEBUG=ON sw' from target/snitch_cluster folder to compile the code.
9. run the code, if it returns 0(more likely just quietly ends), the execution was successfull, otherwise, the return code is the first number that is not equal as the one in results.h

