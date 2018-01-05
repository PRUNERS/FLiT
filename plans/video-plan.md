# FLiT video tutorial plan:

1. Download and install
    - Use without downloading
    - Install in your home directory
    - Install on the system
        - in default location
        - in /opt
        - in /usr/local
2. Review of command-line tool and current capabilities
    - help command
    - init
    - run
    - mention which commands are not yet implemented
3. Initialize with litmus tests
    - demonstrate creation
    - browse an example
        - I like the idea of doing FMACancel, but I can't on my laptop since it
          doesn't have fma capabilities.
        - I could do the FMACancel if I remotely login to another computer such
          as Fractus.
    - run all examples with "--no-timing"
        - State that if we have timing on, it will repeat the tests multiple
          times to get accurate timing numbers.  Running all of these tests one
          after another will take about 5 minutes instead of 3 seconds (on my
          laptop).
        - ./devrun -L
        - ./devrun -L | wc
        - python -c "print('timing lower bound: %s min' % (2.0 * 3 * 43 / 60.0))"
        - Rule of thumb is 0.1 minutes per test is a minimum.  Longer for tests
          that take longer than 2 seconds to run.
    - Run the browsed example only and with timing
4. Write some tests to demonstrate some of the functionalities
    - Write a test that has stochasticity
    - Write a test that has a cuda kernel too
5. Browse the flit-config.toml file and custom.mk
6. Take an actual library and make some litmus tests
    - MFEM library
    - Create a test from scratch from one of the mini applications or one of
      the examples
7. Import results into a database
8. Browse the database
9. Talk about future features
