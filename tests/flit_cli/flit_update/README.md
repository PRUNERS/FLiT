# Tests

## Tests covering `[[compiler]]` section

1. `tst_nocompilers.py`: Test that not specifying the `[[compiler]]` section
   gives the default values
2. `tst_nooptl.py`: Test that specifying a compiler, but not specifying the
   optimization levels gives the default values
3. `tst_noswitches.py`: Test that specifying a compiler, but not specifying the
   switches list gives the default values
4. `tst_onlyprovidedoptlswitches.py`: Test that the provided list of
   optimization levels and switches are used, and nothing more.
5. `tst_onlyprovidedcompilers.py`: Test that by only specifying one compiler,
   only that specified compiler is used
[ ] 6. Test error cases, such as specifying more than one type of a compiler

# Needed tests

1. `FLIT_INC_DIR`, `FLIT_LIB_DIR`, `FLIT_DATA_DIR`, and `FLIT_SCRIPT_DIR`, for
   both from git repository and for an installed FLiT
2. `DEV_CC`, `DEV_OPTL`, and `DEV_SWITCHES` for both provided and non-provided
3. `GT_CC`, `GT_OPTL`, and `GT_SWITCHES` for both provided and non-provided
4. `TEST_RUN_ARGS` from `timing`, `timing_loops`, and `timing_repeats` for both
   provided and non-provided
5. `ENABLE_MPI` and `MPIRUN_ARGS` for both provided and non-provided
6. `HOSTNAME`
7. `UNAME_S`

