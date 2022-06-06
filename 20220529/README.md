Recently, I debugged the code for the interesting paper **Geometric compression of invariant manifolds in neural nets**. And I have learned that how the parameters evolve when the symmetry is in the data. Also, it has obvious connection with the neural tangent kernel. All the codes refer to its original opensourced codes for the paper. Since the style of codes use lot of **yield** instead of return, I'm not used to this style. **And it is strange that the output(ex. dict with label 'final_kernel') is missing, but it is there when I reduce the variable max_wall(from 30000 to 300, alpha.sh with Stripe1LalphaKernel6 and Stripe1LalphaKernel_test1)**. I do not debug the code any more until needs.

1. the .sh file is for the run script
 
2. ntk_compare.py file check the calculation of ntk and the author's method runs very fast
 
3. the show*.py files are for plot

4. the result of ODE*.py is a little weird

