# KmeansTester

## Instructions
- Install the `scikit-learn` python package:
  ```
  python -m pip install scikit-learn
  ```
- place both `kmeanstester.py` and `kmeanstester.c` in the same directory as your C and Python code.
- Add the following lines at the top of your c code:
  ```
  #include "kmeanstester.c"
  #define malloc tester_malloc
  #define calloc tester_calloc
  #define realloc tester_realloc
  ```
  - **REMEMBER TO REMOVE THEM BEFORE SUBMITTING YOUR CODE**
- Recompile your code
- Make sure the directory `kmeans_tester` doesn't exist (the tester uses it)
- Run `python kmeanstester.py` in the same directory as your code.
- Delete the directory `kmeans_tester` after you are done testing.
